use auria_core::{ExecutionOutput, ExecutionState, AuriaResult, Tensor};
use async_trait::async_trait;

#[async_trait]
pub trait CpuBackend: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> AuriaResult<ExecutionOutput>;
}

pub struct CpuExecutionEngine<B: CpuBackend> {
    backend: B,
}

impl<B: CpuBackend> CpuExecutionEngine<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> AuriaResult<ExecutionOutput> {
        self.backend.execute(input, experts, state).await
    }
}
