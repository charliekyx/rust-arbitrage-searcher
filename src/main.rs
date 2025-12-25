//
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
        function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)
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
    token_other: Address,
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

async fn verify_pool(
    client: Arc<SignerMiddleware<Arc<Provider<Ipc>>, LocalWallet>>,
    pair_address: Address,
    router_address: Address,
) -> Result<PoolConfig> {
    let contract = IUniswapV2Pair::new(pair_address, client.clone());
    let token0 = contract.token_0().call().await?;
    let token1 = contract.token_1().call().await?;

    let weth = Address::from_str(WETH_ADDR)?;

    // Identify WETH order and find the other token
    let (order, token_other) = if token0 == weth {
        (TokenOrder::WethFirst, token1)
    } else if token1 == weth {
        (TokenOrder::UsdcFirst, token0)
    } else {
        return Err(anyhow!("Pool must contain WETH"));
    };

    Ok(PoolConfig {
        name: String::new(),
        address: pair_address,
        router: router_address,
        order,
        token_other,
    })
}

// --- Main Entry ---

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("System Starting: Base L2 MEV Bot");

    // 1. Decrypt Configuration
    let config = load_encrypted_config()?;

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
    let password = match env::var("CONFIG_PASS") {
        Ok(p) => p,
        Err(_) => {
            eprint!("Enter Config Password: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            input.trim().to_string()
        }
    };

    let mut file =
        File::open("mev_bot.secure").context("Config file 'mev_bot.secure' not found")?;
    // Cocoon needs mut to update internal state
    let cocoon = Cocoon::new(password.as_bytes());

    let decrypted_bytes = cocoon
        .parse(&mut file)
        .map_err(|e| anyhow!("content decryption error: {:?}", e))?;

    let config: AppConfig = serde_json::from_slice(&decrypted_bytes)
        .map_err(|e| anyhow!("content parse error: {:?}", e))?;

    if config.private_key.is_empty() || config.ipc_path.is_empty() {
        return Err(anyhow!("Decrypted config contains empty fields"));
    }

    info!("Configuration decrypted successfully.");
    Ok(config)
}

// --- Bot Logic ---

async fn run_bot(config: AppConfig) -> Result<()> {
    // 1. Initialize
    // Used in multiple places, using Arc to save resources
    let provider = Arc::new(Provider::<Ipc>::connect_ipc(&config.ipc_path).await?);

    let wallet = LocalWallet::from_str(&config.private_key)?.with_chain_id(8453u64);
    let my_addr = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider.clone(), wallet.clone()));

    let contract_addr: Address = config.contract_address.parse()?;
    let executor = FlashLoanExecutor::new(contract_addr, client.clone());

    // Cross-task survival (GasManager)
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
                pool.name = config.name;
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

    // ==========================================
    // 2. Data Initialization
    // ==========================================
    let reserves = Arc::new(DashMap::new());

    // --- [Step A] Initial Fetch: Warm up the cache ---
    info!("Prefetching reserves for {} pools...", pools.len());
    let mut success_count = 0;
    let start_block = provider.get_block_number().await?.as_u64();

    for pool in &pools {
        let pair = IUniswapV2Pair::new(pool.address, client.clone());
        match pair.get_reserves().call().await {
            Ok((r0, r1, _)) => {
                reserves.insert(pool.address, (U256::from(r0), U256::from(r1), start_block));
                success_count += 1;
            }
            Err(_) => { /* Ignore initial errors */ }
        }
    }
    info!(
        "Reserves initialized: {}/{} pools ready.",
        success_count,
        pools.len()
    );

    // ==========================================
    // 3. Main Arbitrage Loop & Polling
    // ==========================================
    // We removed the tokio::spawn to fix the lifetime (E0597) issue.
    // Instead, we fetch logs INSIDE the main loop directly.

    let nonce_manager = Arc::new(NonceManager::new(provider.clone(), my_addr).await?);

    // Subscribe to new blocks
    let mut stream = client.subscribe_blocks().await?;
    let sync_event_signature =
        H256::from_str("0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1")?;

    info!("Bot Running... Waiting for blocks...");

    loop {
        // Wait for the next block
        let block = match tokio::time::timeout(Duration::from_secs(15), stream.next()).await {
            Ok(Some(b)) => b,
            Ok(None) => return Err(anyhow!("WebSocket Stream Ended")),
            Err(_) => {
                let msg = "Heartbeat Lost: No blocks for 15s";
                send_email(&config, "Heartbeat Lost", msg).await;
                return Err(anyhow!(msg));
            }
        };

        let current_bn = block.number.unwrap(); // This is ethers::types::U64

        if gas_manager.get_loss() >= MAX_DAILY_GAS_LOSS_WEI {
            let msg = format!(
                "Daily Gas Limit Reached ({:.4} ETH).",
                format_ether(gas_manager.get_loss())
            );
            send_email(&config, "Bot Stopping", &msg).await;
            return Err(anyhow!(msg));
        }

        // --- PART 1: Pull Logs & Update Reserves ---
        // Fetch logs for the current block immediately
        let filter = Filter::new()
            .from_block(current_bn)
            .to_block(current_bn)
            .topic0(sync_event_signature);

        if let Ok(logs) = client.get_logs(&filter).await {
            for log in logs {
                if reserves.contains_key(&log.address) {
                    // Optional: Debug log
                    info!("Log Found in Block {}: {:?}", current_bn, log.address);

                    if log.data.len() >= 64 {
                        let r0 = U256::from_big_endian(&log.data[0..32]);
                        let r1 = U256::from_big_endian(&log.data[32..64]);
                        // Update DashMap
                        reserves.insert(log.address, (r0, r1, current_bn.as_u64()));
                    }
                }
            }
        }

        // --- PART 2: Arbitrage Calculation ---
        for i in 0..pools.len() {
            for j in 0..pools.len() {
                if i == j {
                    continue;
                }
                let (pa, pb) = (&pools[i], &pools[j]);

                // Token matching check
                if pa.token_other != pb.token_other {
                    continue;
                }

                // match_count += 1;

                if let (Some(da), Some(db)) = (reserves.get(&pa.address), reserves.get(&pb.address))
                {
                    let (ra0, ra1, bn_a) = *da;
                    let (rb0, rb1, bn_b) = *db;

                    let min_liq = U256::from(100_000_000_000_000_000u128);

                    let weth_a = if pa.order == TokenOrder::WethFirst {
                        ra0
                    } else {
                        ra1
                    };
                    let weth_b = if pb.order == TokenOrder::WethFirst {
                        rb0
                    } else {
                        rb1
                    };

                    if weth_a < min_liq || weth_b < min_liq {
                        continue;
                    }

                    // Stale data check: Ensure logs are recent
                    // Fix: Convert current_bn (U64) to u64 for comparison
                    // if current_bn.as_u64() > bn_a + 3 || current_bn.as_u64() > bn_b + 3 {
                    //     continue;
                    // }

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

                    let (opt_amt, profit_wei) =
                        ternary_search_optimal_amount(ra_in, ra_out, rb_in, rb_out);

                    if profit_wei <= I256::zero() {
                        continue;
                    }
                    let profit_u256 = U256::try_from(profit_wei).unwrap_or_default();

                    // Cost estimation
                    let estimated_gas_limit = U256::from(350_000);
                    let (base_fee, priority_fee) = estimate_eip1559_fees(&provider).await?;
                    let gas_price = base_fee + priority_fee;
                    let estimated_gas_cost_wei = gas_price * estimated_gas_limit;

                    // Target profit
                    let min_net_profit = parse_ether("0.00005")?;
                    let dynamic_threshold = estimated_gas_cost_wei + min_net_profit;

                    if profit_u256 > U256::zero() && profit_u256 <= dynamic_threshold {
                        info!(
                            "Too poor: [{} -> {}] Profit: {} ETH < Cost+Threshold: {} ETH (Gas: {} ETH)",
                            pa.name,
                            pb.name,
                            format_ether(profit_u256),
                            format_ether(dynamic_threshold),
                            format_ether(estimated_gas_cost_wei)
                        );
                    }

                    if profit_u256 > dynamic_threshold {
                        let safe_amt = opt_amt * 99 / 100;
                        let contract_min_profit = dynamic_threshold;

                        info!(
                            "Opp found [{} -> {}]! Profit: {} ETH, Gas Cost: {} ETH. Action: GO",
                            pa.name,
                            pb.name,
                            format_ether(profit_u256),
                            format_ether(estimated_gas_cost_wei)
                        );

                        // Execute Transaction
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

                        // Simulate execution, continue if failed
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
                record.realized_profit = Some(format_ether(exp));
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

    let base_fee = block.base_fee_per_gas.unwrap_or(U256::from(100_000_000));
    let priority_fee = parse_units("0.15", "gwei")?.into();

    Ok((base_fee, priority_fee))
}
