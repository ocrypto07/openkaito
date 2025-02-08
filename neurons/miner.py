import os
import time
import openai
from datetime import datetime
import bittensor as bt
from dotenv import load_dotenv
import asyncio
from openkaito.base.miner import BaseMinerNeuron
from openkaito.protocol import (
    DiscordSearchSynapse,
    SearchSynapse,
    SemanticSearchSynapse,
    StructuredSearchSynapse,
    TextEmbeddingSynapse,
)
from openkaito.utils.embeddings import openai_embeddings_tensor
from openkaito.utils.version import compare_version, get_version

load_dotenv()

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic.
    """

    def __init__(self):
        super(Miner, self).__init__()

    async def forward_text_embedding(self, query: TextEmbeddingSynapse) -> TextEmbeddingSynapse:
        texts = query.texts
        dimensions = query.dimensions

        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),  # Fixed from os.getenv[] to os.getenv()
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=2,  # Reduce retries to avoid long waits
            timeout=5,  # Set a 5-second timeout
        )

        embeddings = openai_embeddings_tensor(
            client, texts, dimensions=dimensions, model="text-embedding-3-large"
        )
        query.results = embeddings.tolist()
        return query

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            "Miner | "
            f"Epoch:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"Rank:{metagraph.R[self.uid]} | "
            f"Trust:{metagraph.T[self.uid]} | "
            f"Consensus:{metagraph.C[self.uid]} | "
            f"Incentive:{metagraph.I[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)

    def check_version(self, query):
        if (
            query.version is not None
            and compare_version(query.version, get_version()) > 0
        ):
            bt.logging.warning(
                f"Received request with version {query.version}, is newer than miner running version {get_version()}. You may need to update the repo and restart the miner."
            )

# This is the main function, which runs the miner.
if __name__ == "__main__":
    miner = Miner()  # Create an instance of Miner
    miner_hotkey = miner.wallet.hotkey.ss58_address
    print(f"My Miner hotkey: {miner_hotkey}")

    async def run_miner():
        await asyncio.sleep(120)  # Wait for 120 seconds before starting
        while True:
            miner.print_info()
            await asyncio.sleep(5)  # Replace 30 seconds with 5 seconds for faster updates

    asyncio.run(run_miner())
