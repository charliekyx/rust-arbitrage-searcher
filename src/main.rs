use anyhow::{anyhow, Context, Result};
use chrono::Local;
use cocoon::Cocoon;
use dashmap::DashMap;
use ethers::{
    prelude::*,
    types::{Address, Eip1559TransactionRequest, H256, I256, U256, U64},
    utils::{format_ether, parse_ether, parse_units},
};
use lettre::{
    message::header::ContentType, transport::smtp::authentication::Credentials, AsyncSmtpTransport,
    AsyncTransport, Message, Tokio1Executor,
};
use serde::{Deserialize, Serialize};
use std::{
    env,
    fs::{self, File, OpenOptions},
    io::Write,
    str::FromStr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tracing::{error, info, warn};

// --- Config Struct ---
#[derive(Serialize, Deserialize, Debug, Clone)]
struct AppConfig {
    private_key: String,
    ipc_path: String,
    contract_address: String,
    smtp_username: String,
    smtp_password: String,
    my_email: String,
}

// --- Constants ---
const WETH_ADDR: &str = "0x4200000000000000000000000000000000000006";
const USDC_ADDR: &str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913";
const MAX_DAILY_GAS_LOSS_WEI: u128 = 20_000_000_000_000_000;
const SLIPPAGE_TOLERANCE_BPS: u64 = 100;

// --- ABI Definitions ---
abigen!(
    FlashLoanExecutor, r#"[function executeArb(uint256 amountToBorrow, address[] targets, bytes[] payloads, uint256 minProfit) external]"#;
    IUniswapV2Router, r#"[function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] path, address to, uint deadline) external returns (uint[] memory amounts)]"#;
   IUniswapV2Pair, r#"[
        function token0() external view returns (address)
        function token1() external view returns (address)
    ]"#
);

#[derive(Debug, Deserialize)]
struct JsonPoolInput {
    name: String,
    pair: String,
    router: String,
}

// --- Data Structures ---
#[derive(Clone, Debug, PartialEq)]
enum TokenOrder {
    UsdcFirst,
    WethFirst,
}

#[derive(Clone, Debug)]
struct PoolConfig {
    name: String,
    address: Address,
    router: Address,
    order: TokenOrder,
    token_other: Address, // [新增] 记录非 WETH 的那个代币地址
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct TradeRecord {
    timestamp: String,
    block_number: u64,
    pool_a: String,
    pool_b: String,
    borrow_amount: String,
    expected_profit: String,
    realized_profit: Option<String>,
    tx_hash: String,
    status: String,
    gas_cost_eth: String,
    error_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct GasState {
    date: String,
    accumulated_loss: u128,
}

struct SharedGasManager {
    accumulated_loss: Mutex<u128>, // Arc默认是不允许修改内部数据的, 保证同一时间只有一个人能修改这个数据
    file_path: String,
}

impl SharedGasManager {
    fn new(path: String) -> Self {
        let loaded = load_gas_state(&path);
        Self {
            accumulated_loss: Mutex::new(loaded.accumulated_loss),
            file_path: path,
        }
    }
    fn add_loss(&self, loss: u128) {
        let mut guard = self.accumulated_loss.lock().unwrap();
        *guard += loss;
        let state = GasState {
            date: Local::now().format("%Y-%m-%d").to_string(),
            accumulated_loss: *guard,
        };
        if let Ok(json) = serde_json::to_string(&state) {
            let _ = fs::write(&self.file_path, json);
        }
        // 锁的周期绑定在 guard 这个变量的生命周期上
        // Rust 编译器会自动调用 guard 的 drop() 方法。
        // 在 drop() 里面，Rust 会自动执行“解锁”操作。
    }
    fn get_loss(&self) -> u128 {
        *self.accumulated_loss.lock().unwrap()
    }
}

struct NonceManager {
    nonce: AtomicU64,
    provider: Arc<Provider<Ipc>>,
    address: Address,
}

impl NonceManager {
    async fn new(provider: Arc<Provider<Ipc>>, address: Address) -> Result<Self> {
        let start_nonce = provider.get_transaction_count(address, None).await?;
        Ok(Self {
            nonce: AtomicU64::new(start_nonce.as_u64()),
            provider,
            address,
        })
    }
    fn get_next(&self) -> U256 {
        U256::from(self.nonce.fetch_add(1, Ordering::SeqCst))
    }
    async fn sync_from_chain(&self) -> Result<()> {
        let on_chain = self
            .provider
            .get_transaction_count(self.address, None)
            .await?;
        self.nonce.store(on_chain.as_u64(), Ordering::SeqCst);
        warn!("Nonce resynced to {}", on_chain);
        Ok(())
    }
}

// Helper function to verify pool and determine token order
// This extracts the logic that was previously hardcoded inside the loop
async fn verify_pool(
    client: Arc<SignerMiddleware<Arc<Provider<Ipc>>, LocalWallet>>,
    pair_address: Address,
    router_address: Address,
) -> Result<PoolConfig> {
    let contract = IUniswapV2Pair::new(pair_address, client.clone());
    let token0 = contract.token_0().call().await?;
    let token1 = contract.token_1().call().await?;

    let weth = Address::from_str(WETH_ADDR)?;

    // 识别 WETH 顺序，同时找出另一个代币是谁
    let (order, token_other) = if token0 == weth {
        (TokenOrder::WethFirst, token1) // token0 是 WETH，那 token1 就是我们要的
    } else if token1 == weth {
        (TokenOrder::UsdcFirst, token0) // token1 是 WETH，那 token0 就是我们要的
    } else {
        return Err(anyhow!("Pool must contain WETH"));
    };

    Ok(PoolConfig {
        name: String::new(),
        address: pair_address,
        router: router_address,
        order,
        token_other, // [新增] 保存代币地址
    })
}
// --- Main Entry ---

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // NOTE: dotenv() has been removed. We now strictly enforce encrypted config.
    info!("System Starting: Base L2 MEV Bot");

    // 1. Decrypt Configuration
    let config = load_encrypted_config()?;

    // Send Startup Email using the decrypted config
    send_email(
        &config,
        "Bot Started",
        "Encrypted configuration loaded successfully.",
    )
    .await;

    loop {
        match run_bot(config.clone()).await {
            Ok(_) => {
                warn!("Main loop finished unexpectedly. Restarting in 5s...");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Err(e) => {
                error!("Critical Crash: {:?}", e);
                send_email(&config, "Bot Crashed", &format!("{:?}", e)).await;
                std::process::exit(1);
            }
        }
    }
}

// --- Helper to load config ---
fn load_encrypted_config() -> Result<AppConfig> {
    // Try to get password from environment variable (for supervisor/docker)
    // or prompt user if interactive.
    let password = match env::var("CONFIG_PASS") {
        Ok(p) => p,
        Err(_) => {
            // If running in background, this will fail, which is intended security.
            // For manual run:
            eprint!("Enter Config Password: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            input.trim().to_string()
        }
    };

    let mut file =
        File::open("mev_bot.secure").context("Config file 'mev_bot.secure' not found")?;
    // Cocoon 需要 mut 来更新内部状态
    let cocoon = Cocoon::new(password.as_bytes());

    let decrypted_bytes = cocoon
        .parse(&mut file)
        .map_err(|e| anyhow!("content decryption error: {:?}", e))?;

    let config: AppConfig = serde_json::from_slice(&decrypted_bytes)
        .map_err(|e| anyhow!("content parse error: {:?}", e))?;

    // Security check: ensure sensitive fields are not empty
    if config.private_key.is_empty() || config.ipc_path.is_empty() {
        return Err(anyhow!("Decrypted config contains empty fields"));
    }

    info!("Configuration decrypted successfully.");
    Ok(config)
}

// --- Bot Logic ---

async fn run_bot(config: AppConfig) -> Result<()> {
    // 1. Initialize using Config Object (NOT env vars)
    // 多处需要这个链接，这里使用arc节省资源
    let provider = Arc::new(Provider::<Ipc>::connect_ipc(&config.ipc_path).await?);

    let wallet = LocalWallet::from_str(&config.private_key)?.with_chain_id(8453u64);
    let my_addr = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider.clone(), wallet.clone()));

    let contract_addr: Address = config.contract_address.parse()?;
    let executor = FlashLoanExecutor::new(contract_addr, client.clone());

    // 跨任务存活(GasManager), 防止后台任务(spawn_tracker)比main后结束生命周期
    let gas_manager = Arc::new(SharedGasManager::new("gas_state.json".to_string()));
    if gas_manager.get_loss() >= MAX_DAILY_GAS_LOSS_WEI {
        let msg = format!(
            "Daily Gas Limit Reached ({:.4} ETH).",
            format_ether(gas_manager.get_loss())
        );
        send_email(&config, "Startup Failed", &msg).await;
        return Err(anyhow!(msg));
    }

    info!("Loading pools from pools.json...");

    let config_content = fs::read_to_string("pools.json")
        .map_err(|e| anyhow!("Failed to read pools.json: {}", e))?;

    let json_configs: Vec<JsonPoolInput> = serde_json::from_str(&config_content)
        .map_err(|e| anyhow!("Failed to parse pools.json: {}", e))?;

    info!(
        "Found {} entries in config file. Verifying on-chain...",
        json_configs.len()
    );

    let mut pools = Vec::new();

    for config in json_configs {
        let pair_address = Address::from_str(&config.pair)?;
        let router_address = Address::from_str(&config.router)?;

        match verify_pool(client.clone(), pair_address, router_address).await {
            Ok(mut pool) => {
                pool.name = config.name; // Set the name from JSON
                info!("Loaded Pool: {} ({:?})", pool.name, pool.address);
                pools.push(pool);
            }
            Err(e) => {
                error!("Failed to verify pool {}: {}", config.name, e);
            }
        }
    }

    if pools.is_empty() {
        return Err(anyhow!("No valid pools loaded. Check pools.json"));
    }

    let usdc = Address::from_str(USDC_ADDR)?;
    let weth = Address::from_str(WETH_ADDR)?;

    // 3. Log Listener
    let reserves = Arc::new(DashMap::new()); // 并发哈希表（Concurrent HashMap）。
                                             //tokio::spawn 里的任务需要“拿走”一个变量的所有权，如果你把 reserves 直接给它，主线程手里就没东西可用了
                                             // r_clone 在后台任务里写入的数据，reserves 在主线程里立马就能读到
    let r_clone = reserves.clone();
    let p_clone = provider.clone();
    let filter = Filter::new()
        .address(pools.iter().map(|p| p.address).collect::<Vec<_>>()) // 只关心被白名单的dex合约地址事件
        .topic0(H256::from_str(
            "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1",
        )?); // 只关心Sync(uint112, uint112)事件， keccak256("Sync(uint112,uint112)")

    tokio::spawn(async move {
        let mut stream = p_clone.subscribe_logs(&filter).await.unwrap();
        while let Some(log) = stream.next().await {
            info!("👂 RX Log from: {:?}", log.address);
            // 在以太坊虚拟机 (EVM) 的日志数据 (data) 中，所有数字通常都会被填充到 32 字节 (256位) 的长度
            // Sync 事件的结构: Sync 事件有两个参数：reserve0 和 reserve1, 总共64字节

            if log.data.len() == 64 {
                if let Ok(d) = ethers::abi::decode(
                    &[
                        ethers::abi::ParamType::Uint(112), //池子中 Token0 的余额（例如 USDC 的数量）
                        ethers::abi::ParamType::Uint(112), // 池子中 Token1 的余额（例如 weth 的数量）
                    ],
                    &log.data,
                ) {
                    let r0 = d[0].clone().into_uint().unwrap();
                    let r1 = d[1].clone().into_uint().unwrap();
                    r_clone.insert(log.address, (r0, r1, log.block_number.unwrap_or_default()));
                }
            }
        }
    });

    let nonce_manager = Arc::new(NonceManager::new(provider.clone(), my_addr).await?);
    let mut stream = client.subscribe_blocks().await?;

    info!("Bot Running...");

    loop {
        let block = match tokio::time::timeout(Duration::from_secs(15), stream.next()).await {
            Ok(Some(b)) => b,                                          // 情况1: 正常收到区块
            Ok(None) => return Err(anyhow!("WebSocket Stream Ended")), // 情况2: 连接被服务器关闭
            Err(_) => {
                // 情况3: 超时了 (心跳丢失)
                let msg = "Heartbeat Lost: No blocks for 15s";
                send_email(&config, "Heartbeat Lost", msg).await;
                return Err(anyhow!(msg));
            }
        };

        let current_bn = block.number.unwrap();

        // ============ hearbeat 检测，后续删除 =================
        // if current_bn.as_u64() % 15 == 0 {
        //     info!(
        //         "Heartbeat: Alive at Block {} | Monitoring {} pools | Gas: {} gwei",
        //         current_bn,
        //         pools.len(),
        //         // 简单的 Gas 估算展示 (Option 处理)
        //         format_ether(block.base_fee_per_gas.unwrap_or_default() * 1_000_000_000) // 简易转换显示，或者直接不显示Gas也行
        //     );
        // }
        // =====================================================

        if gas_manager.get_loss() >= MAX_DAILY_GAS_LOSS_WEI {
            let msg = format!(
                "Daily Gas Limit Reached ({:.4} ETH).",
                format_ether(gas_manager.get_loss())
            );
            send_email(&config, "Bot Stopping", &msg).await;
            return Err(anyhow!(msg));
        }

        for i in 0..pools.len() {
            for j in 0..pools.len() {
                if i == j {
                    continue;
                }
                let (pa, pb) = (&pools[i], &pools[j]);

                // ============ 核心修复：代币匹配检查 ============
                // 如果池子 A 卖的是 DEGEN，池子 B 收的是 USDC，这就不能套利！
                // 必须确保两个池子交易的是同一种“非 WETH”代币。
                if pa.token_other != pb.token_other {
                    continue;
                }

                if let (Some(da), Some(db)) = (reserves.get(&pa.address), reserves.get(&pb.address))
                {
                    // 套利策略：先用 WETH 买 USDC (Pool A)，再用 USDC 买回 WETH (Pool B)

                    let (ra0, ra1, bn_a) = *da;
                    let (rb0, rb1, bn_b) = *db;

                    // 垃圾池过滤：如果池子里的 WETH 少于 0.1 ETH，直接跳过
                    // WETH 精度是 18，0.1 ETH = 10^17 Wei
                    // 定义最小流动性阈值 (0.1 WETH)
                    let min_liq = U256::from(100_000_000_000_000_000u128);

                    // 精准定位 Pool A 的 WETH 余额
                    let weth_a = if pa.order == TokenOrder::WethFirst {
                        ra0
                    } else {
                        ra1
                    };
                    // 精准定位 Pool B 的 WETH 余额
                    let weth_b = if pb.order == TokenOrder::WethFirst {
                        rb0
                    } else {
                        rb1
                    };

                    // 只要任意一个池子的 WETH 余额不足 0.1，直接跳过
                    if weth_a < min_liq || weth_b < min_liq {
                        // 可选：打印一下被过滤的垃圾池，方便确认
                        // info!(
                        //     "Filtering dust pool: {} (WETH: {})",
                        //     pa.name,
                        //     format_ether(weth_a)
                        // );
                        continue;
                    }

                    if current_bn > bn_a + 3 || current_bn > bn_b + 3 {
                        continue;
                    }

                    let (ra_in, ra_out) = if pa.order == TokenOrder::UsdcFirst {
                        (ra1, ra0)
                    } else {
                        (ra0, ra1)
                    };

                    let (rb_in, rb_out) = if pb.order == TokenOrder::UsdcFirst {
                        (rb0, rb1)
                    } else {
                        (rb1, rb0)
                    };

                    // Ternary Search
                    let (opt_amt, profit_wei) =
                        ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);

                    if profit_wei <= I256::zero() {
                        continue;
                    }
                    let profit_u256 = U256::try_from(profit_wei).unwrap_or_default();

                    // 1. 估算 Gas 成本
                    // 闪电贷交易通常消耗 200,000 - 350,000 Gas。为了安全，我们按 350,000 估算。
                    let estimated_gas_limit = U256::from(350_000);
                    let (base_fee, priority_fee) = estimate_eip1559_fees(&provider).await?;
                    let gas_price = base_fee + priority_fee;
                    let estimated_gas_cost_wei = gas_price * estimated_gas_limit;

                    // 2. 设定你的“净利润”目标 (你真正想装进口袋的钱)
                    // 比如赚 0.00005 ETH (约 $0.15) 就愿意跑
                    let min_net_profit = parse_ether("0.00005")?;

                    // 3. 动态计算这就交易需要的“毛利润”阈值
                    let dynamic_threshold = estimated_gas_cost_wei + min_net_profit;

                    if profit_u256 > dynamic_threshold {
                        let safe_amt = opt_amt * 99 / 100;
                        let contract_min_profit = dynamic_threshold;

                        info!(
                            "Opp found [{} -> {}]! Profit: {} ETH, Gas Cost: {} ETH. Action: GO",
                            pa.name, // 买入池
                            pb.name, // 卖出池
                            format_ether(profit_u256),
                            format_ether(estimated_gas_cost_wei)
                        );

                        // Slippage Protection
                        let path_a = vec![weth, usdc];
                        let path_b = vec![usdc, weth];

                        let out_a = get_amount_out_local(safe_amt, ra_in, ra_out);
                        let min_out_a = out_a * (10000 - SLIPPAGE_TOLERANCE_BPS) / 10000;
                        let out_b = get_amount_out_local(out_a, rb_in, rb_out);
                        let min_out_b = out_b * (10000 - SLIPPAGE_TOLERANCE_BPS) / 10000;
                        let deadline = U256::from(
                            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() + 60,
                        );

                        let payload1 = IUniswapV2Router::new(pa.router, client.clone())
                            .swap_exact_tokens_for_tokens(
                                safe_amt,
                                min_out_a,
                                path_a,
                                contract_addr,
                                deadline,
                            )
                            .calldata()
                            .ok_or(anyhow!("Payload1 failed"))?;

                        let payload2 = IUniswapV2Router::new(pb.router, client.clone())
                            .swap_exact_tokens_for_tokens(
                                out_a,
                                min_out_b,
                                path_b,
                                contract_addr,
                                deadline,
                            )
                            .calldata()
                            .ok_or(anyhow!("Payload2 failed"))?;

                        let tx_call = executor.execute_arb(
                            safe_amt,
                            vec![pa.router, pb.router],
                            vec![payload1, payload2],
                            contract_min_profit,
                        );

                        if tx_call.call().await.is_err() {
                            continue;
                        }

                        let (base_fee, priority_fee) = estimate_eip1559_fees(&provider).await?;
                        let max_fee = base_fee * 120 / 100 + priority_fee;
                        let nonce = nonce_manager.get_next();

                        let tx_req = Eip1559TransactionRequest::new()
                            .to(contract_addr)
                            .data(tx_call.calldata().unwrap())
                            .gas(600_000)
                            .max_fee_per_gas(max_fee)
                            .max_priority_fee_per_gas(priority_fee)
                            .nonce(nonce);

                        match client.send_transaction(tx_req, None).await {
                            Ok(pending) => {
                                info!("Tx Sent: {:?}", pending.tx_hash());
                                spawn_tracker(
                                    provider.clone(),
                                    gas_manager.clone(),
                                    config.clone(),
                                    pending.tx_hash(),
                                    current_bn.as_u64(),
                                    pa.name.clone(),
                                    pb.name.clone(),
                                    safe_amt,
                                    contract_min_profit,
                                );
                            }
                            Err(e) => {
                                error!("Send Error: {:?}", e);
                                let _ = nonce_manager.sync_from_chain().await;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn spawn_tracker(
    provider: Arc<Provider<Ipc>>,
    gas: Arc<SharedGasManager>,
    config: AppConfig,
    hash: H256,
    bn: u64,
    p_a: String,
    p_b: String,
    amt: U256,
    exp: U256,
) {
    tokio::spawn(async move {
        let mut receipt = None;
        for _ in 0..15 {
            if let Ok(Some(r)) = provider.get_transaction_receipt(hash).await {
                receipt = Some(r);
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        if let Some(r) = receipt {
            let used = r.gas_used.unwrap_or_default();
            let price = r.effective_gas_price.unwrap_or_default();
            let cost = used * price;

            let mut record = TradeRecord {
                timestamp: Local::now().to_rfc3339(),
                block_number: bn,
                pool_a: p_a,
                pool_b: p_b,
                borrow_amount: format_ether(amt),
                expected_profit: format_ether(exp),
                realized_profit: None,
                tx_hash: format!("{:?}", hash),
                status: "Pending".to_string(),
                gas_cost_eth: format_ether(cost),
                error_reason: None,
            };

            if r.status != Some(U64::from(1)) {
                record.status = "Revert".to_string();
                gas.add_loss(cost.as_u128());
                send_email(
                    &config,
                    "Revert",
                    &format!("Tx: {:?}\nLoss: {} ETH", hash, format_ether(cost)),
                )
                .await;
            } else {
                record.status = "Success".to_string();
                record.realized_profit = Some(format_ether(exp)); // Simplified
                send_email(
                    &config,
                    "Success",
                    &format!("Tx: {:?}\nProfit: {} ETH", hash, format_ether(exp)),
                )
                .await;
            }
            log_trade(&record);
        }
    });
}

// --- Email Helper (Now uses Config) ---
async fn send_email(config: &AppConfig, subject: &str, body: &str) {
    if config.smtp_username.is_empty() {
        return;
    }
    let email = Message::builder()
        .from(config.smtp_username.parse().unwrap())
        .to(config.my_email.parse().unwrap())
        .subject(subject)
        .header(ContentType::TEXT_PLAIN)
        .body(body.to_string())
        .unwrap();

    let creds = Credentials::new(config.smtp_username.clone(), config.smtp_password.clone());
    let mailer: AsyncSmtpTransport<Tokio1Executor> =
        AsyncSmtpTransport::<Tokio1Executor>::relay("smtp.gmail.com")
            .unwrap()
            .credentials(creds)
            .build();
    let _ = mailer.send(email).await;
}

// --- Math & Logging ---
// 套利利润曲线是一个倒 U 型抛物线，所以必须用三分法找极值点

fn ternary_search_optimal_amount(
    ra_in: U256,
    ra_out: U256,
    rb_in: U256,
    rb_out: U256,
) -> (U256, I256) {
    let mut low = U256::zero();
    let mut high = ra_in;
    for _ in 0..50 {
        if high <= low {
            break;
        }
        let diff = high - low;
        let m1 = low + diff / 3;
        let m2 = high - diff / 3;
        if simulate_profit(m1, ra_in, ra_out, rb_in, rb_out)
            < simulate_profit(m2, ra_in, ra_out, rb_in, rb_out)
        {
            low = m1;
        } else {
            high = m2;
        }
    }
    let best = (low + high) / 2;
    (best, simulate_profit(best, ra_in, ra_out, rb_in, rb_out))
}

fn simulate_profit(amt_in: U256, ra_in: U256, ra_out: U256, rb_in: U256, rb_out: U256) -> I256 {
    let amt_mid = get_amount_out_local(amt_in, ra_in, ra_out);
    let amt_final = get_amount_out_local(amt_mid, rb_in, rb_out);
    I256::from_raw(amt_final) - I256::from_raw(amt_in)
}

fn get_amount_out_local(amount_in: U256, reserve_in: U256, reserve_out: U256) -> U256 {
    if amount_in.is_zero() || reserve_in.is_zero() || reserve_out.is_zero() {
        return U256::zero();
    }
    let amount_in_with_fee = amount_in * 997;
    let numerator = amount_in_with_fee * reserve_out;
    let denominator = (reserve_in * 1000) + amount_in_with_fee;
    numerator / denominator
}

fn log_trade(record: &TradeRecord) {
    if let Ok(j) = serde_json::to_string(record) {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open("trades.jsonl")
            .unwrap();
        let _ = writeln!(f, "{}", j);
    }
}

fn load_gas_state(path: &str) -> GasState {
    let today = Local::now().format("%Y-%m-%d").to_string();
    if let Ok(c) = fs::read_to_string(path) {
        if let Ok(s) = serde_json::from_str::<GasState>(&c) {
            if s.date == today {
                return s;
            }
        }
    }
    GasState {
        date: today,
        accumulated_loss: 0,
    }
}

async fn estimate_eip1559_fees(provider: &Provider<Ipc>) -> Result<(U256, U256)> {
    let block = provider
        .get_block(BlockNumber::Latest)
        .await?
        .ok_or_else(|| anyhow!("No block"))?;

    // Base 链的基础费
    let base_fee = block.base_fee_per_gas.unwrap_or(U256::from(100_000_000));

    // 动态调整优先费：如果是在抢机会，给高一点，比如 0.15 - 0.5 gwei
    // 这里简单给一个比地板价稍高的值，确保快速打包
    let priority_fee = parse_units("0.15", "gwei")?.into();

    Ok((base_fee, priority_fee))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethers::utils::parse_ether;

    fn to_u256(n: &str) -> U256 {
        parse_ether(n).unwrap()
    }

    #[test]
    fn test_simulate_profit_profitable() {
        // 借入 ETH。
        // Pool A (第一站): 价格要高 (贵)，要在这里卖 ETH。
        // Pool A: 100 ETH, 110,000 USDC (价格 1100)
        let ra_in = to_u256("100"); // ETH Reserve
        let ra_out = to_u256("110000"); // USDC Reserve

        // Pool B (第二站): 价格要低 (便宜)，我们要在这里买回 ETH。
        // Pool B: 100 ETH, 100,000 USDC (价格 1000)
        let rb_in = to_u256("100000"); // USDC Reserve (调低，变便宜)
        let rb_out = to_u256("100"); // ETH Reserve

        // 投入 1 ETH
        let amount_in = to_u256("1");

        let profit = simulate_profit(amount_in, ra_in, ra_out, rb_in, rb_out);

        // 预期逻辑：1 ETH 在 A 换成 ~1100 USDC，去 B 换回 ~1.1 ETH，利润 ~0.1 ETH
        println!("Profit for 1 ETH input: {:?}", profit);
        assert!(
            profit > I256::zero(),
            "Profit should be positive, but got {:?}",
            profit
        );
    }

    #[test]
    fn test_ternary_search_optimality() {
        // Pool A (高价卖出): 价格 2000
        let ra_in = to_u256("100");
        let ra_out = to_u256("200000");

        // Pool B (低价买入): 价格 1000
        let rb_in = to_u256("100000");
        let rb_out = to_u256("100");

        // 寻找最佳投入金额
        let (best_amt, max_profit) = ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);

        println!("Best Amount: {:?}, Max Profit: {:?}", best_amt, max_profit);

        // 断言：必须找到正收益
        assert!(best_amt > U256::zero());
        assert!(max_profit > I256::zero());

        // 验证“山顶”逻辑：
        // 比最佳金额少投一点点，或者多投一点点，利润都应该变少
        let one_wei = U256::one();
        let profit_at_best = simulate_profit(best_amt, ra_in, ra_out, rb_in, rb_out);

        // 如果 best_amt 很大，测试 -1 wei 和 +1 wei
        if best_amt > one_wei {
            let profit_at_less = simulate_profit(best_amt - one_wei, ra_in, ra_out, rb_in, rb_out);
            assert!(
                profit_at_best >= profit_at_less,
                "Peak check failed: best < less"
            );
        }

        let profit_at_more = simulate_profit(best_amt + one_wei, ra_in, ra_out, rb_in, rb_out);
        assert!(
            profit_at_best >= profit_at_more,
            "Peak check failed: best < more"
        );
    }

    #[test]
    fn test_unprofitable_scenario() {
        // --- 亏损场景设置 ---
        // Pool A (卖出站): 价格很低，贱卖。
        // 100 ETH : 100,000 USDC -> 价格 1000 USDC/ETH
        let ra_in = to_u256("100");
        let ra_out = to_u256("100000");

        // Pool B (买回站): 价格很高，贵买。
        // 100 ETH : 110,000 USDC -> 价格 1100 USDC/ETH
        let rb_in = to_u256("110000");
        let rb_out = to_u256("100");

        // 1. 测试单笔计算：投入 1 ETH 会发生什么？
        let amount_in = to_u256("1");
        let profit = simulate_profit(amount_in, ra_in, ra_out, rb_in, rb_out);

        println!("Loss for 1 ETH input: {:?}", profit);

        // 断言：利润必须是负数 (I256 < 0)
        assert!(
            profit < I256::zero(),
            "Should be losing money but got positive profit!"
        );

        // 2. 测试搜索算法：在这种情况下，机器人应该建议我们投多少钱？
        // 既然怎么投都亏，最优策略应该是“不投” (0 ETH)。
        let (best_amt, max_profit) = ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);

        println!(
            "Best Amount in bad market: {:?}, Profit: {:?}",
            best_amt, max_profit
        );

        // 断言：最优投入金额应该是 0 (或者非常接近 0)，利润应该是 0 (不亏就是赚)
        assert!(
            max_profit <= I256::zero(),
            "Safety Check Failed: Bot thinks it can make money!"
        );
        let dust_threshold = to_u256("0.000001"); // 允许 0.000001 ETH 的误差
        assert!(
            best_amt < dust_threshold,
            "Bot suggested risking too much money!"
        );
    }

    #[test]
    fn test_ternary_search_convergence() {
        let ra_in = to_u256("100");
        let ra_out = to_u256("200000");
        let rb_in = to_u256("100000");
        let rb_out = to_u256("100");

        // 运行两次搜索，验证结果是否一致
        let (amt1, profit1) = ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);
        let (amt2, profit2) = ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);

        assert_eq!(amt1, amt2, "Search results should be deterministic");
        assert_eq!(profit1, profit2, "Profit should be deterministic");

        // 验证精度（搜索范围应该收敛到 < 0.001 ETH）
        let precision = to_u256("0.001");
        let profit_at_plus = simulate_profit(amt1 + precision, ra_in, ra_out, rb_in, rb_out);
        let profit_at_minus = simulate_profit(amt1 - precision, ra_in, ra_out, rb_in, rb_out);

        let tolerance = profit1.abs() / 100; // 1% 误差
        assert!(
            (profit1 - profit_at_plus).abs() <= tolerance,
            "Search precision insufficient"
        );
        assert!(
            (profit1 - profit_at_minus).abs() <= tolerance,
            "Search precision insufficient"
        );
    }
    #[test]
    fn test_uniswap_fee_calculation() {
        // 简化场景：1:1 价格，无价格冲击
        let reserve_in = to_u256("1000000"); // 1M ETH
        let reserve_out = to_u256("1000000"); // 1M USDC
        let amount_in = to_u256("1"); // 1 ETH

        let amount_out = get_amount_out_local(amount_in, reserve_in, reserve_out);

        // 预期输出 = 1 * 0.997 = 0.997 ETH（扣除 0.3% 手续费）
        let expected = to_u256("0.997");

        println!("Amount Out: {:?}", amount_out);
        println!("Expected: {:?}", expected);

        // 允许 0.1% 的误差
        let diff = if amount_out > expected {
            amount_out - expected
        } else {
            expected - amount_out
        };
        assert!(diff < to_u256("0.001"), "Fee calculation error");
    }

    #[test]
    fn test_excessive_input() {
        let ra_in = to_u256("100");
        let ra_out = to_u256("110000");
        let rb_in = to_u256("100000");
        let rb_out = to_u256("100");

        // 投入 1000 ETH（远超储备量）
        let amount_in = to_u256("1000");
        let profit = simulate_profit(amount_in, ra_in, ra_out, rb_in, rb_out);

        // 预期：利润为负（因为价格冲击太大）
        println!("Profit for excessive input: {:?}", profit);
        assert!(
            profit < I256::zero(),
            "Should lose money due to high slippage"
        );
    }
    #[test]
    fn test_zero_input() {
        let ra_in = to_u256("100");
        let ra_out = to_u256("110000");
        let rb_in = to_u256("100000");
        let rb_out = to_u256("100");

        let amount_in = U256::zero();
        let profit = simulate_profit(amount_in, ra_in, ra_out, rb_in, rb_out);

        assert_eq!(profit, I256::zero());
    }
    #[test]
    fn fuzz_test_ternary_search() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let ra_in = U256::from(rng.gen_range(1..1_000_000));
            let ra_out = U256::from(rng.gen_range(1..1_000_000));
            let rb_in = U256::from(rng.gen_range(1..1_000_000));
            let rb_out = U256::from(rng.gen_range(1..1_000_000));

            let (best_amt, max_profit) =
                ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);

            // 验证不会 panic
            assert!(best_amt >= U256::zero());
        }
    }
}
