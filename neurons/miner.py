import os
import time
import openai
import bittensor as bt
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from openkaito.base.miner import BaseMinerNeuron
from openkaito.protocol import (
    TextEmbeddingSynapse,
)
from openkaito.utils.embeddings import openai_embeddings_tensor
from openkaito.utils.version import compare_version, get_version

load_dotenv()

class OpenAIClient:
    """Handles OpenAI API calls safely with error handling."""
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=2,  
            timeout=5,
        )

    async def get_embeddings(self, texts, dimensions):
        """Fetches embeddings from OpenAI, handling errors gracefully."""
        try:
            return openai_embeddings_tensor(
                self.client, texts, dimensions=dimensions, model="text-embedding-3-large"
            ).tolist()
        except Exception as e:
            bt.logging.error(f"OpenAI Embedding Error: {e}")
            return None

class Miner(BaseMinerNeuron):
    """Custom Miner for Bittensor running OpenKaito."""

    def __init__(self):
        super().__init__()
        self.openai_client = OpenAIClient()  

    async def forward_text_embedding(self, query: TextEmbeddingSynapse) -> TextEmbeddingSynapse:
        """Handles text embedding requests."""
        embeddings = await self.openai_client.get_embeddings(query.texts, query.dimensions)
        if embeddings:
            query.results = embeddings
        return query  

    def log_miner_status(self):
        """Logs miner status with timestamps."""
        metagraph = self.metagraph
        self.uid = metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            f"[{datetime.utcnow()}] Miner Status | "
            f"Epoch: {self.step} | UID: {self.uid} | Block: {self.block} | "
            f"Stake: {metagraph.S[self.uid]:.4f} | Rank: {metagraph.R[self.uid]:.4f} | "
            f"Trust: {metagraph.T[self.uid]:.4f} | Consensus: {metagraph.C[self.uid]:.4f} | "
            f"Incentive: {metagraph.I[self.uid]:.4f} | Emission: {metagraph.E[self.uid]:.4f}"
        )
        bt.logging.info(log)

    def check_version(self, query):
        """Warns if the miner is outdated."""
        if query.version and compare_version(query.version, get_version()) > 0:
            bt.logging.warning(f"Newer version detected: {query.version} > {get_version()}")

async def run_miner(miner):
    """Runs the miner with periodic logging."""
    await asyncio.sleep(10)  # Start delay reduced from 120 to 10 seconds
    while True:
        miner.log_miner_status()
        await asyncio.sleep(10)  # Faster updates with 10-second intervals

if __name__ == "__main__":
    miner = Miner()
    print(f"My Miner hotkey: {miner.wallet.hotkey.ss58_address}")
    try:
        asyncio.run(run_miner(miner))
    except KeyboardInterrupt:
        print("Miner shutting down gracefully.")
