# ========================
# Stage 1: Builder
# ========================
FROM rust:1.83-slim-bookworm as builder

WORKDIR /usr/src/app

# Install build dependencies (OpenSSL is required for ethers-rs)
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# --- Caching Trick Start ---
# Create a dummy project to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
COPY Cargo.toml Cargo.lock ./
# Build specific dependencies only
RUN cargo build --release
# Remove the dummy binary
RUN rm -f target/release/deps/mev_bot*
# --- Caching Trick End ---

# Copy actual source code
COPY src ./src

# Build the actual application
RUN cargo build --release

# ========================
# Stage 2: Runtime
# ========================
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
# ca-certificates is needed for HTTPS (Flashbots/RPC)
# libssl3 is needed for ethers-rs
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /usr/src/app/target/release/mev-bot /app/mev-bot

# Use a non-root user for security (Optional but recommended)
RUN useradd -m mevuser
USER mevuser

# Run the binary
CMD ["./mev-bot"]