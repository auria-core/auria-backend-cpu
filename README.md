# auria-backend-cpu

CPU execution backend for AURIA Runtime Core.

## Overview

Implements tensor operations and expert execution on CPU hardware.

## Usage

```rust
use auria_backend_cpu::{CpuBackend, CpuExecutionEngine};

let engine = CpuExecutionEngine::new(backend);
let output = engine.execute(input, experts, state).await?;
```
