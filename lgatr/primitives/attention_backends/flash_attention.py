"""original flash-attention backend."""

try:
    from flash_attn import flash_attn_varlen_func
except ModuleNotFoundError as err:
    raise ImportError(
        "flash-attn is not installed. Run 'pip install lgatr[flash-attention]'."
    ) from err

# There is no fancy docs website, so one has to check the source code for the interface:
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py
attention = flash_attn_varlen_func
