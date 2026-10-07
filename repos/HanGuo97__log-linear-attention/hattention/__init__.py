__version__ = "0.0.1"

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from hattention.configuration_hattention import HAttentionConfig
from hattention.modeling_hattention import HAttentionForCausalLM, HAttentionModel

from hattention.configuration_mamba2mlp import Mamba2MLPConfig
from hattention.modeling_mamba2mlp import Mamba2ForCausalLM, Mamba2Model

from hattention.configuration_h_gated_deltanet import HGatedDeltaNetConfig
from hattention.modeling_h_gated_deltanet import HGatedDeltaNetForCausalLM, HGatedDeltaNetModel

AutoConfig.register(HAttentionConfig.model_type, HAttentionConfig, exist_ok=False)
AutoModel.register(HAttentionConfig, HAttentionModel, exist_ok=False)
AutoModelForCausalLM.register(HAttentionConfig, HAttentionForCausalLM, exist_ok=False)

AutoConfig.register(Mamba2MLPConfig.model_type, Mamba2MLPConfig, exist_ok=False)
AutoModel.register(Mamba2MLPConfig, Mamba2Model, exist_ok=False)
AutoModelForCausalLM.register(Mamba2MLPConfig, Mamba2ForCausalLM, exist_ok=False)

AutoConfig.register(HGatedDeltaNetConfig.model_type, HGatedDeltaNetConfig, exist_ok=False)
AutoModel.register(HGatedDeltaNetConfig, HGatedDeltaNetModel, exist_ok=False)
AutoModelForCausalLM.register(HGatedDeltaNetConfig, HGatedDeltaNetForCausalLM, exist_ok=False)


__all__ = ["HGatedDeltaNetConfig", "HGatedDeltaNetForCausalLM", "HGatedDeltaNetModel",
               "HAttentionConfig",     "HAttentionForCausalLM",     "HAttentionModel",
                "Mamba2MLPConfig",         "Mamba2ForCausalLM",         "Mamba2Model"]
