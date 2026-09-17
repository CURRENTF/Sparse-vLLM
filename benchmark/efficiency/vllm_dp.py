"""Single-node AsyncLLM adapter for globally coordinated fixed request batches."""

import asyncio
import copy


class VLLMDPBatch:
    def __init__(self, engine_kwargs):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.dp_size = int(engine_kwargs["data_parallel_size"])
        self.batch_id = 0
        self.runner = asyncio.Runner()

        async def initialize():
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))

        try:
            self.engine = self.runner.run(initialize())
        except BaseException:
            self.runner.close()
            raise

    async def _generate(self, prompts, sampling_params):
        from vllm.sampling_params import RequestOutputKind

        params = copy.copy(sampling_params)
        params.output_kind = RequestOutputKind.FINAL_ONLY
        batch_id = self.batch_id
        self.batch_id += 1

        async def consume(index, prompt):
            output = None
            async for item in self.engine.generate(
                prompt, params, request_id=f"dp-{batch_id}-{index}",
                data_parallel_rank=index % self.dp_size,
            ):
                output = item
            if output is None or not output.finished:
                raise RuntimeError(f"DP request {index} did not finish")
            return output

        # TaskGroup cancels sibling requests on failure; results retain trace order.
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(consume(i, prompt))
                     for i, prompt in enumerate(prompts)]
        return [task.result() for task in tasks]

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return self.runner.run(self._generate(prompts, sampling_params))

    def close(self):
        try:
            self.engine.shutdown()
        finally:
            self.runner.close()
