use anyhow::{Context, Result};
use ethers::{
    abi::AbiDecode,
    prelude::*,
    utils::{format_ether, parse_ether},
};
use ethers_flashbots::{BundleRequest, FlashbotsMiddleware};
use std::{env, str::FromStr, sync::Arc};
use tracing::{error, info, warn};
use url::Url;

// ABI Updated with new Events and Errors
abigen!(
    BundleExecutor,
    r#"[
        function executeArb(address _target, bytes calldata _payload, uint256 _amountIn, uint256 _minProfit, uint256 _minerBribe) external payable returns (uint256)
        function setExecutor(address _newExecutor) external
        function executor() external view returns (address)
        
        event ArbExecuted(address indexed target, uint256 amountIn, uint256 profit, uint256 minerBribe)
        
        error InsufficientProfit(uint256 required, uint256 available)
        error TransferFailed()
        error ETHTransferFailed()
        error NotExecutor()
    ]"#
);

#[derive(Clone, Debug)]
struct SimulationParams {
    target: Address,
    payload: Bytes,
    amount_in: U256,
    min_profit: U256,
    miner_bribe: U256,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    dotenv::dotenv().ok();
    
    info!("Starting MEV Bot [Secure Executor Mode]");

    let rpc_url = env::var("RPC_URL").context("RPC_URL missing")?;
    let private_key = env::var("PRIVATE_KEY").context("PRIVATE_KEY missing")?;
    let contract_addr = env::var("CONTRACT_ADDRESS").context("CONTRACT_ADDRESS missing")?;
    
    let provider = Provider::<Ws>::connect(&rpc_url).await?;
    let wallet = LocalWallet::from_str(&private_key)?.with_chain_id(1u64);
    let bot_address = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider.clone(), wallet.clone()));

    let fb_client = FlashbotsMiddleware::new(
        provider.clone(),
        Url::parse("https://relay.flashbots.net")?,
        wallet.clone(),
    );

    let contract = BundleExecutor::new(Address::from_str(&contract_addr)?, client.clone());

    // Pre-flight check
    info!("Verifying permissions...");
    let authorized_executor = contract.executor().call().await?;
    if authorized_executor != bot_address {
        error!("PERMISSION DENIED: Bot address {:?} is not the authorized executor.", bot_address);
        return Err(anyhow::anyhow!("Bot wallet not authorized"));
    }
    info!("Permission verified.");

    let mut stream = client.subscribe_blocks().await?;
    
    while let Some(block) = stream.next().await {
        let block_number = block.number.unwrap();
        
        // Mock Opportunity Logic
        let mut params = SimulationParams {
            target: Address::random(), 
            payload: Bytes::from(vec![0u8; 64]),
            amount_in: parse_ether("1.0")?,
            min_profit: parse_ether("0.05")?,
            miner_bribe: parse_ether("0.2")?,
        };

        let mut retry_count = 0;
        let max_retries = 1; 

        loop {
            // Build call
            let call = contract.execute_arb(
                params.target,
                params.payload.clone(),
                params.amount_in,
                params.min_profit,
                params.miner_bribe,
            );

            match call.call().await {
                Ok(net_profit) => {
                    info!("Sim Success! Net Profit: {} ETH. Sending Bundle...", format_ether(net_profit));
                    
                    let tx_req = call.tx;
                    let mut bundle = BundleRequest::new()
                        .push_transaction(tx_req)
                        .set_block(block_number + 1)
                        .set_simulation_block(block_number)
                        .set_simulation_timestamp(0);

                    match fb_client.send_bundle(&bundle).await {
                        Ok(resp) => info!("Bundle sent. Hash: {:?}", resp.bundle_hash),
                        Err(e) => error!("Flashbots error: {:?}", e),
                    }
                    break;
                }
                Err(contract_err) => {
                    if let Some(decoded) = decode_error(&contract_err) {
                        match decoded {
                            BundleExecutorErrors::InsufficientProfit(err_data) => {
                                if retry_count >= max_retries { break; }

                                let gross_profit = err_data.available;
                                if gross_profit <= params.min_profit { break; }

                                let max_affordable_bribe = gross_profit - params.min_profit;
                                let safe_bribe = max_affordable_bribe * 99 / 100;

                                info!("Adjusting Bribe: {} -> {}", format_ether(params.miner_bribe), format_ether(safe_bribe));
                                params.miner_bribe = safe_bribe;
                                retry_count += 1;
                            },
                            BundleExecutorErrors::ETHTransferFailed(_) => {
                                // Critical: The miner (coinbase) refused the payment or call failed
                                error!("CRITICAL: ETH Transfer to Miner failed (DoS risk or gas issue).");
                                break;
                            },
                            BundleExecutorErrors::TransferFailed(_) => {
                                error!("Critical: ERC20 Transfer failed.");
                                break;
                            },
                            BundleExecutorErrors::NotExecutor(_) => {
                                error!("CRITICAL: Permission revoked.");
                                return Err(anyhow::anyhow!("Permission revoked"));
                            },
                            _ => {
                                error!("Unhandled contract error: {:?}", decoded);
                                break;
                            }
                        }
                    } else {
                        error!("RPC Simulation error: {}", contract_err);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

fn decode_error(err: &ContractError<SignerMiddleware<Provider<Ws>, LocalWallet>>) -> Option<BundleExecutorErrors> {
    let revert_bytes = match err {
        ContractError::Revert(b) => b,
        _ => return None,
    };
    BundleExecutorErrors::decode(revert_bytes).ok()
}