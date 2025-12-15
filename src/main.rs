use anyhow::{anyhow, Context, Result};
use chrono::Local;
use cocoon::Cocoon;
use dashmap::DashMap;
use ethers::{
    abi::AbiEncode,
    prelude::*,
    types::{Address, H256, U256, I256, U64, Eip1559TransactionRequest},
};
use ethers::utils::format_ether;
use lettre::{
    message::header::ContentType, transport::smtp::authentication::Credentials, AsyncSmtpTransport,
    Message, Tokio1Executor, Transport,
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
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
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
    IUniswapV2Pair, r#"[function token0() external view returns (address)]"#
);

// --- Data Structures ---
#[derive(Clone, Debug, PartialEq)]
enum TokenOrder { UsdcFirst, WethFirst }

#[derive(Clone, Debug)]
struct PoolConfig {
    name: String,
    address: Address,
    router: Address,
    order: TokenOrder,
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
    accumulated_loss: Mutex<u128>,
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
        Ok(Self { nonce: AtomicU64::new(start_nonce.as_u64()), provider, address })
    }
    fn get_next(&self) -> U256 { U256::from(self.nonce.fetch_add(1, Ordering::SeqCst)) }
    async fn sync_from_chain(&self) -> Result<()> {
        let on_chain = self.provider.get_transaction_count(self.address, None).await?;
        self.nonce.store(on_chain.as_u64(), Ordering::SeqCst);
        warn!("🔄 Nonce resynced to {}", on_chain);
        Ok(())
    }
}

type ReservesMap = Arc<DashMap<Address, (U256, U256, U64)>>;

// --- Main Entry ---

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    // NOTE: dotenv() has been removed. We now strictly enforce encrypted config.
    info!("🛡️ System Starting: Base L2 MEV Bot (Encrypted Mode)");

    // 1. Decrypt Configuration
    let config = load_encrypted_config()?;
    
    // Send Startup Email using the decrypted config
    send_email(&config, "🟢 Bot Started", "Encrypted configuration loaded successfully.").await;

    loop {
        match run_bot(config.clone()).await {
            Ok(_) => {
                warn!("Main loop finished unexpectedly. Restarting in 5s...");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Err(e) => {
                error!("🔥 Critical Crash: {:?}", e);
                send_email(&config, "🔥 Bot Crashed", &format!("{:?}", e)).await;
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

    let mut file = File::open("mev_bot.secure").context("Config file 'mev_bot.secure' not found")?;
    let cocoon = Cocoon::new(password.as_bytes());

    let decrypted_bytes = cocoon.parse(&mut file)
    .map_err(|e| anyhow!("解密失败: {:?}", e))?;

   let config: AppConfig = serde_json::from_slice(&decrypted_bytes)
    .map_err(|e| anyhow!("配置解析失败 (JSON格式错误): {:?}", e))?;
    
    // Security check: ensure sensitive fields are not empty
    if config.private_key.is_empty() || config.ipc_path.is_empty() {
        return Err(anyhow!("Decrypted config contains empty fields"));
    }
    
    info!("✅ Configuration decrypted successfully.");
    Ok(config)
}

// --- Bot Logic ---

async fn run_bot(config: AppConfig) -> Result<()> {
    // 1. Initialize using Config Object (NOT env vars)
    let provider = Arc::new(Provider::<Ipc>::connect_ipc(&config.ipc_path).await?);
    
    let wallet = LocalWallet::from_str(&config.private_key)?.with_chain_id(8453u64);
    let my_addr = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider.clone(), wallet.clone()));
    
    let contract_addr: Address = config.contract_address.parse()?;
    let executor = FlashLoanExecutor::new(contract_addr, client.clone());

    let gas_manager = Arc::new(SharedGasManager::new("gas_state.json".to_string()));
    if gas_manager.get_loss() >= MAX_DAILY_GAS_LOSS_WEI {
        let msg = format!("Daily Gas Limit Reached ({:.4} ETH).", format_ether(gas_manager.get_loss().into()));
        send_email(&config, "🛑 Startup Failed", &msg).await;
        return Err(anyhow!(msg));
    }

    // 2. Whitelist Setup
    let raw_whitelist = vec![
        ("BaseSwap", "0x696b47741D53c8ec7A65FE537F7D2141F91671F6", "0x2948acbbc8795267e62a1220683a48e718b52585"),
        ("SushiSwap", "0x905dfcd5649217c42684f23958568e533c711aa3", "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506"),
        ("AlienBase", "0x1927a412c019488a101f3f6197b91d293222dc38", "0x8c1A3cF8f83074169FE5D7aD50B978e1cd6b37c7"),
        ("SwapBased", "0xc56e632b7337351658428135832a2253842c6725", "0xD4a7FEbD52efda82d6f8acE24908aE0aa5b4f956"),
    ];

    let usdc = Address::from_str(USDC_ADDR)?;
    let weth = Address::from_str(WETH_ADDR)?;
    let mut pools = Vec::new();

    info!("🔍 Verifying Pools...");
    for (name, pair, router) in raw_whitelist {
        let pair_addr = Address::from_str(pair)?;
        let contract = IUniswapV2Pair::new(pair_addr, client.clone());
        let token0 = contract.token0().call().await?;
        let order = if token0 == usdc { TokenOrder::UsdcFirst } 
                    else if token0 == weth { TokenOrder::WethFirst }
                    else { continue; };
        pools.push(PoolConfig { name: name.to_string(), address: pair_addr, router: Address::from_str(router)?, order });
    }

    // 3. Log Listener
    let reserves = Arc::new(DashMap::new());
    let r_clone = reserves.clone();
    let p_clone = provider.clone();
    let filter = Filter::new()
        .address(pools.iter().map(|p| p.address).collect::<Vec<_>>())
        .topic0(H256::from_str("0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1")?);

    tokio::spawn(async move {
        let mut stream = p_clone.subscribe_logs(&filter).await.unwrap();
        while let Some(log) = stream.next().await {
            if log.data.len() == 64 {
                if let Ok(d) = ethers::abi::decode(&[ethers::abi::ParamType::Uint(112), ethers::abi::ParamType::Uint(112)], &log.data) {
                    let r0 = d[0].clone().into_uint().unwrap();
                    let r1 = d[1].clone().into_uint().unwrap();
                    r_clone.insert(log.address, (r0, r1, log.block_number.unwrap_or_default()));
                }
            }
        }
    });

    let nonce_manager = Arc::new(NonceManager::new(provider.clone(), my_addr).await?);
    let mut stream = client.subscribe_blocks().await?;
    let mut last_hb = Instant::now();

    info!("🚀 Bot Running...");

    while let Some(block) = stream.next().await {
        last_hb = Instant::now();
        let current_bn = block.number.unwrap();

        if gas_manager.get_loss() >= MAX_DAILY_GAS_LOSS_WEI {
            let msg = format!("Daily Gas Limit Reached ({:.4} ETH).", format_ether(gas_manager.get_loss().into()));
            send_email(&config, "🛑 Bot Stopping", &msg).await;
            return Err(anyhow!(msg));
        }

        for i in 0..pools.len() {
            for j in 0..pools.len() {
                if i == j { continue; }
                let (pa, pb) = (&pools[i], &pools[j]);
                
                if let (Some(da), Some(db)) = (reserves.get(&pa.address), reserves.get(&pb.address)) {
                    let (ra0, ra1, bn_a) = *da;
                    let (rb0, rb1, bn_b) = *db;
                    if current_bn > bn_a + 3 || current_bn > bn_b + 3 { continue; }

                    let (ra_in, ra_out) = if pa.order == TokenOrder::UsdcFirst { (ra1, ra0) } else { (ra0, ra1) };
                    let (rb_in, rb_out) = if pb.order == TokenOrder::UsdcFirst { (rb0, rb1) } else { (rb1, rb0) };

                    // Ternary Search
                    let (opt_amt, profit_wei) = ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);
                    
                    if profit_wei <= I256::zero() { continue; }
                    let profit_u256 = U256::try_from(profit_wei).unwrap_or_default();
                    let min_profit = parse_ether("0.002")?;

                    if profit_u256 > min_profit {
                        let safe_amt = opt_amt * 99 / 100;
                        let safe_profit = profit_u256 * 95 / 100;

                        info!("💡 Opp: {}->{}. Profit: {}", pa.name, pb.name, format_ether(safe_profit));

                        // Slippage Protection
                        let path_a = vec![weth, usdc];
                        let path_b = vec![usdc, weth];
                        
                        let out_a = get_amount_out_local(safe_amt, ra_in, ra_out);
                        let min_out_a = out_a * (10000 - SLIPPAGE_TOLERANCE_BPS) / 10000;
                        let out_b = get_amount_out_local(out_a, rb_in, rb_out);
                        let min_out_b = out_b * (10000 - SLIPPAGE_TOLERANCE_BPS) / 10000;
                        let deadline = U256::from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() + 60);

                        let payload1 = IUniswapV2Router::new(pa.router, client.clone()).swap_exact_tokens_for_tokens(safe_amt, min_out_a, path_a, contract_addr, deadline).calldata()?;
                        let payload2 = IUniswapV2Router::new(pb.router, client.clone()).swap_exact_tokens_for_tokens(out_a, min_out_b, path_b, contract_addr, deadline).calldata()?;

                        let tx_call = executor.execute_arb(safe_amt, vec![pa.router, pb.router], vec![payload1, payload2], safe_profit);

                        if tx_call.call().await.is_err() { continue; }

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
                                info!("🚀 Tx Sent: {:?}", pending.tx_hash());
                                spawn_tracker(provider.clone(), gas_manager.clone(), config.clone(), pending.tx_hash(), current_bn.as_u64(), pa.name.clone(), pb.name.clone(), safe_amt, safe_profit);
                            },
                            Err(e) => {
                                error!("❌ Send Error: {:?}", e);
                                let _ = nonce_manager.sync_from_chain().await;
                            }
                        }
                    }
                }
            }
        }

        if last_hb.elapsed() > Duration::from_secs(15) {
            send_email(&config, "⚠️ Heartbeat Lost", "Node connection unstable").await;
            return Err(anyhow!("Heartbeat lost"));
        }
    }
    Ok(())
}

fn spawn_tracker(
    provider: Arc<Provider<Ipc>>,
    gas: Arc<SharedGasManager>,
    config: AppConfig,
    hash: H256,
    bn: u64,
    p_a: String, p_b: String,
    amt: U256, exp: U256
) {
    tokio::spawn(async move {
        let mut receipt = None;
        for _ in 0..15 {
             if let Ok(Some(r)) = provider.get_transaction_receipt(hash).await {
                receipt = Some(r); break;
             }
             tokio::time::sleep(Duration::from_secs(1)).await;
        }
        
        if let Some(r) = receipt {
            let used = r.gas_used.unwrap_or_default();
            let price = r.effective_gas_price.unwrap_or_default();
            let cost = used * price;
            
            let mut record = TradeRecord {
                timestamp: Local::now().to_rfc3339(), block_number: bn, pool_a: p_a, pool_b: p_b,
                borrow_amount: format_ether(amt), expected_profit: format_ether(exp), realized_profit: None,
                tx_hash: format!("{:?}", hash), status: "Pending".to_string(), gas_cost_eth: format_ether(cost), error_reason: None,
            };

            if r.status != Some(U64::from(1)) {
                record.status = "Revert".to_string();
                gas.add_loss(cost.as_u128());
                send_email(&config, "❌ Revert", &format!("Tx: {:?}\nLoss: {} ETH", hash, format_ether(cost))).await;
            } else {
                record.status = "Success".to_string();
                record.realized_profit = Some(format_ether(exp)); // Simplified
                send_email(&config, "✅ Success", &format!("Tx: {:?}\nProfit: {} ETH", hash, format_ether(exp))).await;
            }
            log_trade(&record);
        }
    });
}

// --- Email Helper (Now uses Config) ---
async fn send_email(config: &AppConfig, subject: &str, body: &str) {
    if config.smtp_username.is_empty() { return; }
    let email = Message::builder()
        .from(config.smtp_username.parse().unwrap())
        .to(config.my_email.parse().unwrap())
        .subject(subject)
        .header(ContentType::TEXT_PLAIN)
        .body(body.to_string())
        .unwrap();

    let creds = Credentials::new(config.smtp_username.clone(), config.smtp_password.clone());
    let mailer: AsyncSmtpTransport<Tokio1Executor> = AsyncSmtpTransport::<Tokio1Executor>::relay("smtp.gmail.com")
        .unwrap().credentials(creds).build();
    let _ = mailer.send(email).await;
}

// --- Math & Logging ---
fn ternary_search_optimal_amount(ra_in: U256, ra_out: U256, rb_in: U256, rb_out: U256) -> (U256, I256) {
    let mut low = U256::zero();
    let mut high = ra_in;
    for _ in 0..50 {
        if high <= low { break; }
        let diff = high - low;
        let m1 = low + diff / 3;
        let m2 = high - diff / 3;
        if simulate_profit(m1, ra_in, ra_out, rb_in, rb_out) < simulate_profit(m2, ra_in, ra_out, rb_in, rb_out) { low = m1; } else { high = m2; }
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
    if amount_in.is_zero() || reserve_in.is_zero() || reserve_out.is_zero() { return U256::zero(); }
    let amount_in_with_fee = amount_in * 997;
    let numerator = amount_in_with_fee * reserve_out;
    let denominator = (reserve_in * 1000) + amount_in_with_fee;
    numerator / denominator
}

fn log_trade(record: &TradeRecord) {
    if let Ok(j) = serde_json::to_string(record) {
        let mut f = OpenOptions::new().create(true).append(true).open("trades.jsonl").unwrap();
        let _ = writeln!(f, "{}", j);
    }
}

fn load_gas_state(path: &str) -> GasState {
    let today = Local::now().format("%Y-%m-%d").to_string();
    if let Ok(c) = fs::read_to_string(path) {
        if let Ok(s) = serde_json::from_str::<GasState>(&c) {
            if s.date == today { return s; }
        }
    }
    GasState { date: today, accumulated_loss: 0 }
}

async fn estimate_eip1559_fees(provider: &Provider<Ipc>) -> Result<(U256, U256)> {
    let block = provider.get_block(BlockNumber::Latest).await?.ok_or_else(|| anyhow!("No block"))?;
    let base = block.base_fee_per_gas.unwrap_or(U256::from(100_000_000));
    let prio = parse_units("0.1", "gwei")?.into();
    Ok((base, prio))
}

// fn simulate_profit(amt_in: U256, ra_in: U256, ra_out: U256, rb_in: U256, rb_out: U256) -> I256 {
//     let amt_mid = get_amount_out_local(amt_in, ra_in, ra_out);
//     let amt_final = get_amount_out_local(amt_mid, rb_in, rb_out);
//     I256::from_raw(amt_final) - I256::from_raw(amt_in)
// }


// // Uniswap V2 的 getAmountOut 公式（0.3% 手续费)
// fn get_amount_out_local(amount_in: U256, reserve_in: U256, reserve_out: U256) -> U256 {
//     let amount_in_with_fee = amount_in * 997;
//     let numerator = amount_in_with_fee * reserve_out;
//     let denominator = (reserve_in * 1000) + amount_in_with_fee;
//     numerator / denominator
// }

// 1. 高风险：AtomicU64 只能存储 u64 类型，但 GasState.accumulated_loss 是 u128，可能导致溢出！
// 🔴 严重问题：如果 Gas 亏损超过 u64::MAX（约 18.4 ETH），程序会出错。
// 💡 修复建议：改用 AtomicU128（需要 nightly Rust）或使用 Mutex<u128>。

// 2.  将环境变量加密

// 3. log.data.len() >= 32 不够严谨：应该检查是否恰好 64 字节（两个 uint112）。



// 4.使用二分搜索找到最优借款金额
// ⚠️ 风险点：

// 安全系数 98%：可能导致实际利润低于预期。
// unwrap_or_default()：如果 profit 是负数，try_from 会失败，返回 0，导致误判。

// 5. 构造两笔 Swap 的 calldata。
// ⚠️ 风险点：

// amountOutMin = 0：没有滑点保护，如果价格波动，可能亏损！
// deadline = U256::MAX：交易永不过期，可能被延迟打包。


// 💡 优化建议：

// 设置合理的 amountOutMin（如 95% 的预期输出）。
// 设置 deadline（如当前时间 + 60 秒）。


// 💡 优化建议：

// 增加滑点保护（如 95% ~ 99% 动态调整）。
// 检查 profit 是否为正数。


// 6. 先用 call() 模拟执行，成功后再发送真实交易。
// ⚠️ 风险点：

// Gas Limit 固定为 500,000：可能不够用（尤其是多跳套利）。
// Priority Fee 固定为 0.1 Gwei：在高竞争环境下，交易可能被延迟或失败。
// Nonce 管理：如果交易失败，Nonce 不会回退，导致后续交易全部失败！


// 💡 优化建议：

// 动态估算 Gas（estimate_gas()）。
// 根据链上 Gas 价格动态调整 Priority Fee。
// 增加 Nonce 追踪机制，失败时重置。


// 7. 这不是标准的二分搜索！它通过比较 mid 和 mid + 1% 来判断方向，但这种方法不精确。
// 💡 优化建议：

// 使用 三分搜索（Ternary Search）或 牛顿法（Newton's Method）求解最优点。
// 或者直接用数学公式求解（Uniswap V2 的最优套利量有闭式解）。


// 8 风险点：

// 每次调用都写文件，频繁 I/O 会拖慢性能。
// 忽略写入失败（let _ = fs::write(...)），可能导致数据丢失。


// 9.0 合约
// _expectedHash 冲突（合约）⚠️ 中支持多笔并发交易

// 10: Nonce 管理缺陷（🔴 严重）
// 如果交易失败，Nonce 不会回退，导致后续交易全部失败。
// 修复建议合理：增加 Nonce 追踪机制，失败时重置。


// 💡 优化建议：

// 使用缓冲区，每隔 N 次或 N 秒才写入一次。
// 增加错误日志，记录写入失败的情况。

// 11. 如果 targets 包含恶意合约，可能导致重入攻击。
// 修复建议实用：使用白名单验证 targets。

// 12/ 缺少对 I256 负数处理的分析， 如果 profit 是负数，try_from 会失败，返回 0，导致误判。

// 13 缺少对业务逻辑的深入分析问题 3：未分析"三明治攻击"风险MEV Bot 最大的风险之一是被其他 Bot 三明治攻击（Sandwich Attack）：
// 前置交易（Front-run）：攻击者在 Bot 的交易前插入一笔交易，推高价格。
// 后置交易（Back-run）：攻击者在 Bot 的交易后插入一笔交易，低价买回。
// 改进建议：
// 🔴 严重风险：Bot 的交易可能被三明治攻击（Sandwich Attack）：

// 没有使用 Flashbots/Private RPC：交易会进入公开的 Mempool，攻击者可以看到并抢跑。
// amountOutMin = 0：攻击者可以推高价格，导致 Bot 亏损。


// 未分析"区块重组"风险在 Base 链（或任何 PoS 链）上，区块重组（Reorg）可能导致交易失效：
// Bot 在区块 N 发送交易。
// 区块 N 被重组，交易消失。
// Bot 的 Nonce 已经递增，导致后续交易失败。
// 改进建议：
// ⚠️ 中风险：区块重组（Reorg）可能导致交易失效：

// Nonce 管理缺陷：如果交易在 Reorg 中消失，Nonce 不会回退。
// 储备量数据过期：Reorg 后，储备量可能已经变化。

// 💡 防御建议：

// 监听 Reorg 事件：使用 provider.watch_blocks() 检测 Reorg，重置 Nonce。
// 增加确认深度：只在交易被确认（如 3 个区块后）才更新 Gas 亏损。

// ⚠️ 3. 缺少对测试与监控的建议问题 5：缺少单元测试与集成测试代码中没有任何测试，这在生产环境中是极其危险的。改进建议：
// 🔴 严重缺陷：代码中没有任何测试，这在处理资金的场景中是不可接受的。
// 💡 测试建议：

// 单元测试：

// 测试 calculate_optimal_amount 的正确性（使用已知的储备量和预期结果）。
// 测试 get_amount_out_local 是否与链上的 getAmountOut 一致。


// 集成测试：

// 使用 Foundry 或 Hardhat 在本地 Fork 测试网上模拟套利场景。
// 测试合约的 executeArb 是否正确执行闪电贷和 Swap。


// 模拟测试：

// 使用历史区块数据回测（Backtesting），评估策略的盈利能力。



// 问题 6：缺少监控与告警代码中只有邮件通知，但缺少实时监控和告警机制。改进建议：
// ⚠️ 中风险：缺少实时监控和告警机制，可能导致以下问题：

// Gas 亏损超限后才发现：应该在接近限额时提前告警。
// 交易失败率过高：应该监控成功率，及时调整策略。
// 节点连接断开：应该监控 IPC 连接状态，自动重连。

// 💡 监控建议：

// 使用 Prometheus + Grafana 监控关键指标（Gas 亏损、成功率、延迟等）。
// 使用 PagerDuty 或 Slack 实时告警。
// 记录所有交易到数据库（如 PostgreSQL），便于后续分析。



// 💡 防御建议：

// 使用 Flashbots Protect 或 Private RPC（如 Alchemy、Infura 的私有交易服务）。
// 设置合理的 amountOutMin，确保滑点在可接受范围内。
// 增加 Gas Price 竞争机制：如果检测到竞争交易，动态提高 Gas Price。



// do not understand

// IUniswapV2Router 这个是什么意思

// 0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1 这是什么

//    tokio::spawn(async move {
//     let mut stream = provider_clone.subscribe_logs(&filter).await.unwrap();
//     while let Some(log) = stream.next().await {
//         if log.data.len() >= 32 {
//              if let Ok(d) = ethers::abi::decode(&[ethers::abi::ParamType::Uint(112), ethers::abi::ParamType::Uint(112)], &log.data) {
//                  let r0 = d[0].clone().into_uint().unwrap();
//                  let r1 = d[1].clone().into_uint().unwrap();
//                  reserves_clone.insert(log.address, (r0, r1, log.block_number.unwrap_or_default()));
//              }
//         }
//     }
// });
