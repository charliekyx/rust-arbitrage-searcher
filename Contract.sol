// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface IWETH is IERC20 {
    function deposit() external payable;
    function withdraw(uint256) external;
}

contract BundleExecutor is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    IWETH public immutable WETH;
    address public executor;

    // Events for off-chain monitoring
    event ArbExecuted(
        address indexed target,
        uint256 amountIn,
        uint256 profit,
        uint256 minerBribe
    );

    // Custom errors
    error InsufficientProfit(uint256 expected, uint256 actual);
    error TransferFailed();
    error NotExecutor();
    error ETHTransferFailed();

    modifier onlyExecutor() {
        if (msg.sender != executor && msg.sender != owner()) {
            revert NotExecutor();
        }
        _;
    }

    constructor(address _weth) Ownable(msg.sender) {
        WETH = IWETH(_weth);
    }

    receive() external payable {}

    function setExecutor(address _newExecutor) external onlyOwner {
        executor = _newExecutor;
    }

    /// @notice Executes arb with security enhancements
    function executeArb(
        address _target,
        bytes calldata _payload,
        uint256 _amountIn,
        uint256 _minProfit,
        uint256 _minerBribe
    ) external payable onlyExecutor nonReentrant returns (uint256 profit) {
        uint256 balanceBefore = WETH.balanceOf(address(this));

        // Optimization: Only approve if allowance is insufficient
        if (WETH.allowance(address(this), _target) < _amountIn) {
            // SafeERC20 forceApprove or standard approve with max uint
            WETH.forceApprove(_target, type(uint256).max);
        }

        // Interaction: External Call
        // Security: We trust the executor to strictly define the payload
        (bool success, bytes memory reason) = _target.call(_payload);
        if (!success) {
            assembly {
                revert(add(reason, 32), mload(reason))
            }
        }

        uint256 balanceAfter = WETH.balanceOf(address(this));
        
        // Profit Check
        if (balanceAfter <= balanceBefore) {
            revert InsufficientProfit(balanceBefore, balanceAfter);
        }
        uint256 grossProfit = balanceAfter - balanceBefore;

        if (grossProfit < _minProfit) {
            revert InsufficientProfit(_minProfit, grossProfit);
        }

        // Bribe Payment: Use call instead of transfer to support contract-based validators
        if (_minerBribe > 0) {
            if (address(this).balance < _minerBribe) {
                WETH.withdraw(_minerBribe);
            }
            
            (bool sent, ) = block.coinbase.call{value: _minerBribe}("");
            if (!sent) {
                revert ETHTransferFailed();
            }
        }

        profit = grossProfit - _minerBribe;

        emit ArbExecuted(_target, _amountIn, profit, _minerBribe);
    }

    function withdraw(address _token) external onlyOwner {
        if (_token == address(0)) {
            (bool sent, ) = payable(msg.sender).call{value: address(this).balance}("");
            if (!sent) revert ETHTransferFailed();
        } else {
            IERC20(_token).safeTransfer(msg.sender, IERC20(_token).balanceOf(address(this)));
        }
    }
}