// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "openzeppelin-contracts/contracts/access/Ownable.sol";
import "openzeppelin-contracts/contracts/token/ERC20/IERC20.sol";
import "openzeppelin-contracts/contracts/token/ERC20/utils/SafeERC20.sol";

// --- 定义 Uniswap V3 Router 接口 ---
interface ISwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }

    function exactInputSingle(
        ExactInputSingleParams calldata params
    ) external payable returns (uint256 amountOut);
}

interface IFlashLoanRecipient {
    function receiveFlashLoan(
        IERC20[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external;
}

interface IVault {
    function flashLoan(
        IFlashLoanRecipient recipient,
        IERC20[] memory tokens,
        uint256[] memory amounts,
        bytes memory userData
    ) external;
}

contract FlashLoanExecutor is IFlashLoanRecipient, Ownable {
    using SafeERC20 for IERC20;

    // Base 链上的 Balancer Vault 地址 (通常不变，但请核实)
    address private constant BALANCER_VAULT =
        0xBA12222222228d8Ba445958a75a0704d566BF2C8;
    // WETH 地址 (Base)
    address private constant WETH = 0x4200000000000000000000000000000000000006;

    address public executor;

    // --- 结构体定义 ---
    struct SwapStep {
        address router; // V3 Router 地址
        address tokenIn;
        address tokenOut;
        uint24 fee; // 费率: 500 (0.05%), 3000 (0.3%), 10000 (1%)
    }

    struct ArbParams {
        uint256 borrowAmount;
        SwapStep[] steps; // 交易路径
        uint256 minProfit;
    }

    error InsufficientProfit(uint256 balanceAfter, uint256 required);
    error NotBalancer();
    error OnlyExecutorOrOwner();

    constructor() Ownable(msg.sender) {}

    function setExecutor(address _executor) external onlyOwner {
        executor = _executor;
    }

    modifier onlyExecutorOrOwner() {
        if (msg.sender != executor && msg.sender != owner()) {
            revert OnlyExecutorOrOwner();
        }
        _;
    }

    // 授权工具：必须给所有用到的 V3 Router 授权 WETH 和其他中间代币
    function approveToken(
        address token,
        address spender,
        uint256 amount
    ) external onlyOwner {
        IERC20(token).approve(spender, amount);
    }

    receive() external payable {}

    // --- 执行入口 ---
    function executeArb(
        uint256 borrowAmount,
        SwapStep[] calldata steps,
        uint256 minProfit
    ) external onlyExecutorOrOwner {
        // 打包参数传递给回调
        bytes memory userData = abi.encode(
            ArbParams({
                borrowAmount: borrowAmount,
                steps: steps,
                minProfit: minProfit
            })
        );

        IERC20[] memory tokens = new IERC20[](1);
        tokens[0] = IERC20(WETH);
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = borrowAmount;

        // 发起 Balancer 闪电贷
        IVault(BALANCER_VAULT).flashLoan(
            IFlashLoanRecipient(address(this)),
            tokens,
            amounts,
            userData
        );
    }

    // --- 闪电贷回调 ---
    function receiveFlashLoan(
        IERC20[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external override {
        if (msg.sender != BALANCER_VAULT) revert NotBalancer();

        ArbParams memory params = abi.decode(userData, (ArbParams));

        uint256 repayAmount = amounts[0] + feeAmounts[0];
        uint256 balanceBefore = IERC20(WETH).balanceOf(address(this));

        // 当前持有的 Token 数量 (初始为借来的 WETH)
        uint256 currentAmount = amounts[0];

        // --- 循环执行 V3 Swaps ---
        for (uint256 i = 0; i < params.steps.length; i++) {
            SwapStep memory step = params.steps[i];

            // 构造 V3 参数
            ISwapRouter.ExactInputSingleParams memory swapParams = ISwapRouter
                .ExactInputSingleParams({
                    tokenIn: step.tokenIn,
                    tokenOut: step.tokenOut,
                    fee: step.fee,
                    recipient: address(this),
                    deadline: block.timestamp,
                    amountIn: currentAmount,
                    amountOutMinimum: 0, // 我们在 Rust 端算好了，这里设 0 以避免轻微滑点导致 Revert
                    sqrtPriceLimitX96: 0
                });

            // 执行交换，更新 currentAmount
            currentAmount = ISwapRouter(step.router).exactInputSingle(
                swapParams
            );
        }

        // --- 利润检查 ---
        uint256 balanceAfter = IERC20(WETH).balanceOf(address(this));
        uint256 required = balanceBefore + feeAmounts[0] + params.minProfit;

        if (balanceAfter < required) {
            revert InsufficientProfit(balanceAfter, required);
        }

        // 还款
        IERC20(WETH).safeTransfer(BALANCER_VAULT, repayAmount);

        // 提走利润
        uint256 profit = IERC20(WETH).balanceOf(address(this));
        if (profit > 0) {
            IERC20(WETH).safeTransfer(owner(), profit);
        }
    }

    function withdraw(address token) external onlyOwner {
        if (token == address(0)) {
            payable(msg.sender).transfer(address(this).balance);
        } else {
            IERC20(token).safeTransfer(
                msg.sender,
                IERC20(token).balanceOf(address(this))
            );
        }
    }
}
