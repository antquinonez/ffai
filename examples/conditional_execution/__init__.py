from unittest.mock import MagicMock, patch

from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient
from ffai.config import Config
from ffai.FFAI import FFAI

mock_config = MagicMock(spec=Config)
mock_config.paths = MagicMock()
mock_config.paths.ffai_data = "/tmp/ffai_conditional_example"

mock_client = MagicMock(spec=FFLiteLLMClient)
mock_client.model = "mistral/mistral-small-latest"
mock_client.get_conversation_history.return_value = []
mock_client.set_conversation_history = MagicMock()
mock_client.clear_conversation = MagicMock()
mock_client.last_usage = None
mock_client.last_cost_usd = 0.0

with patch("ffai.FFAI.get_config", return_value=mock_config):
    ffai = FFAI(mock_client)

print(f"FFAI initialized (mocked client, model={mock_client.model})")
print(f"History: {len(ffai.history)} entries")
