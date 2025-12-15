// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

// Balancer 接口
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

    // Base Mainnet Addresses
    address private constant BALANCER_VAULT = 0xBA12222222228d8Ba445958a75a0704d566BF2C8;
    address private constant WETH = 0x4200000000000000000000000000000000000006;

    // 安全锁：防止回调被未授权的交易触发
    bytes32 private _expectedHash;

    struct ArbParams {
        address[] targets;
        bytes[] payloads;
        uint256 minProfit;
    }

    error InsufficientProfit(uint256 profit, uint256 required);
    error NotBalancer();
    error UntrustedInitiator();
    error CallFailed(uint256 index, bytes reason);

    constructor() Ownable(msg.sender) {}

    receive() external payable {}

    // --- 1. Bot 入口 ---
    function executeArb(
        uint256 amountToBorrow,
        address[] calldata targets,
        bytes[] calldata payloads,
        uint256 minProfit
    ) external onlyOwner {
        // 计算本次任务的哈希指纹，作为“暗号”
        _expectedHash = keccak256(abi.encode(targets, payloads, minProfit));

        bytes memory userData = abi.encode(ArbParams({
            targets: targets,
            payloads: payloads,
            minProfit: minProfit
        }));

        IERC20[] memory tokens = new IERC20[](1);
        tokens[0] = IERC20(WETH);

        uint256[] memory amounts = new uint256[](1);
        amounts[0] = amountToBorrow;

        // 触发闪电贷
        IVault(BALANCER_VAULT).flashLoan(
            IFlashLoanRecipient(address(this)),
            tokens,
            amounts,
            userData
        );
    }

    // --- 2. Balancer 回调 ---
    function receiveFlashLoan(
        IERC20[] memory tokens,
        uint256[] memory amounts,
        uint256[] memory feeAmounts,
        bytes memory userData
    ) external override {
        if (msg.sender != BALANCER_VAULT) revert NotBalancer();

        ArbParams memory params = abi.decode(userData, (ArbParams));

        // 安全校验：验证暗号
        bytes32 incomingHash = keccak256(abi.encode(params.targets, params.payloads, params.minProfit));
        if (incomingHash != _expectedHash) revert UntrustedInitiator();
        
        // 清除暗号，防止重放
        delete _expectedHash;

        uint256 amountBorrowed = amounts[0];
        uint256 fee = feeAmounts[0];
        uint256 amountToRepay = amountBorrowed + fee;

        uint256 balanceBefore = IERC20(WETH).balanceOf(address(this));

        // 执行交易逻辑
        for (uint256 i = 0; i < params.targets.length; i++) {
            (bool success, bytes memory reason) = params.targets[i].call(params.payloads[i]);
            if (!success) {
                // 如果任意一步失败，直接 Revert，不浪费后续计算
                 assembly {
                    revert(add(reason, 32), mload(reason))
                }
            }
        }

        uint256 balanceAfter = IERC20(WETH).balanceOf(address(this));

        // 利润检查
        if (balanceAfter < balanceBefore + fee + params.minProfit) {
             revert InsufficientProfit(balanceAfter - balanceBefore, fee + params.minProfit);
        }

        // 还款
        IERC20(WETH).safeTransfer(BALANCER_VAULT, amountToRepay);

        // 利润归集给 Owner (或者暂时留在合约里省 Gas，这里选择自动提现)
        uint256 profit = balanceAfter - balanceBefore - fee;
        if (profit > 0) {
            IERC20(WETH).safeTransfer(owner(), profit);
        }
    }

    // 紧急提款
    function withdraw(address token) external onlyOwner {
        if (token == address(0)) {
            payable(msg.sender).transfer(address(this).balance);
        } else {
            IERC20(token).safeTransfer(msg.sender, IERC20(token).balanceOf(address(this)));
        }
    }
}