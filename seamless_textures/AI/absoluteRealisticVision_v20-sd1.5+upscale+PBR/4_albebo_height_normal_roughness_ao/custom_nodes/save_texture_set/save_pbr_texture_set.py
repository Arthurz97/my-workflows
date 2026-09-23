from comfy_api.latest import ComfyExtension, IO, UI
from comfy.cli_args import args
from typing_extensions import override
import os
import re
import torch
import folder_paths

from comfy_extras.nodes_images import (
    _encode_image,
    _save_avif,
    inject_png_metadata,
    inject_exr_metadata,
)


def _build_filename(prefix, counter, padding, suffix, position):
    """Monta o nome final concatenando prefixo, contador e sufixo.

    Nenhum separador é inserido automaticamente — o separador deve vir do
    prefixo ou do sufixo digitado pelo usuário.
    """
    counter_str = f"{counter:0{padding}d}"
    if position == "start":
        return f"{counter_str}{prefix}{suffix}"
    elif position == "middle":
        return f"{prefix}{counter_str}{suffix}"
    else:  # end
        return f"{prefix}{suffix}{counter_str}"


def _find_max_counter(output_dir, prefix, padding, position, extension):
    """Encontra o maior contador já usado, montando uma regex baseada na
    posição do contador e no padding configurado.

    O contador é compartilhado entre todos os sufixos, então a busca ignora
    o sufixo e casa apenas a estrutura do contador (padding dígitos).

    Exemplos (prefix='Texture', padding=5, extension='png'):
        start:  00001Texture...png
        middle: Texture00001...png
        end:    Texture...00001.png
    """
    if not os.path.isdir(output_dir):
        return 1

    p = re.escape(prefix)
    ext = re.escape(extension)
    d = f"\\d{{{padding}}}"  # exatamente `padding` dígitos

    if position == "start":
        pattern = re.compile(rf"^({d}){p}.*\.{ext}$")
    elif position == "middle":
        pattern = re.compile(rf"^{p}({d}).*\.{ext}$")
    else:  # end
        pattern = re.compile(rf"^{p}.*?({d})\.{ext}$")

    max_counter = 0
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            try:
                max_counter = max(max_counter, int(match.group(1)))
            except ValueError:
                pass

    return max_counter + 1


class SavePBRTextureSet(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SavePBRTextureSet",
            search_aliases=[
                "save texture set", "pbr saver", "texture set saver",
                "salvar texturas", "save pbr",
            ],
            display_name="Save PBR Texture Set",
            description=(
                "Saves multiple texture maps with a shared filename prefix, "
                "per-map suffixes, and a configurable counter position."
            ),
            category="image",
            is_output_node=True,
            inputs=[
                IO.String.Input(
                    "filename_prefix",
                    default="Texture",
                    tooltip=(
                        "Base filename shared by all texture maps in this set. "
                        "The separator (e.g. '_') must be included here or in "
                        "the suffix — this node adds no separators automatically. "
                        "May include formatting tokens such as %date:yyyy-MM-dd%. "
                        "Supports subfolders (e.g. 'textures/MedievalStoneWall')."
                    ),
                ),
                IO.Combo.Input(
                    "counter_position",
                    options=["start", "middle", "end"],
                    default="middle",
                    tooltip=(
                        "Where to place the auto-incrementing counter.\n"
                        "start:  00001<prefix><suffix>\n"
                        "middle: <prefix>00001<suffix>\n"
                        "end:    <prefix><suffix>00001"
                    ),
                ),
                IO.Int.Input(
                    "counter_padding",
                    default=5,
                    min=1,
                    max=10,
                    advanced=True,
                    tooltip="Number of digits in the counter.",
                ),
                # --- Slot 1 ---
                IO.Image.Input("image_1"),
                IO.String.Input("suffix_1", default="_albedo",
                                tooltip="Suffix for map 1 (include separator if needed)."),
                # --- Slot 2 ---
                IO.Image.Input("image_2", optional=True),
                IO.String.Input("suffix_2", default="_normal",
                                tooltip="Suffix for map 2 (include separator if needed)."),
                # --- Slot 3 ---
                IO.Image.Input("image_3", optional=True),
                IO.String.Input("suffix_3", default="_height",
                                tooltip="Suffix for map 3 (include separator if needed)."),
                # --- Slot 4 ---
                IO.Image.Input("image_4", optional=True),
                IO.String.Input("suffix_4", default="_roughness",
                                tooltip="Suffix for map 4 (include separator if needed)."),
                # --- Slot 5 ---
                IO.Image.Input("image_5", optional=True),
                IO.String.Input("suffix_5", default="_ao",
                                tooltip="Suffix for map 5 (include separator if needed)."),
                # --- Formato (mesmas opções do SaveImageAdvanced nativo) ---
                IO.DynamicCombo.Input(
                    "format",
                    options=[
                        IO.DynamicCombo.Option("png", [
                            IO.Combo.Input(
                                "bit_depth",
                                options=["8-bit", "16-bit"],
                                default="16-bit",
                                advanced=True,
                            ),
                            IO.Combo.Input(
                                "input_color_space",
                                options=["sRGB"],
                                default="sRGB",
                                advanced=True,
                            ),
                        ]),
                        IO.DynamicCombo.Option("exr", [
                            IO.Combo.Input(
                                "bit_depth",
                                options=["32-bit float"],
                                default="32-bit float",
                                advanced=True,
                            ),
                            IO.Combo.Input(
                                "input_color_space",
                                options=["sRGB", "HDR", "linear"],
                                default="sRGB",
                                advanced=True,
                            ),
                        ]),
                        IO.DynamicCombo.Option("avif", [
                            IO.Combo.Input(
                                "bit_depth",
                                options=["auto", "8-bit YUV420", "10-bit YUV420"],
                                default="auto",
                                advanced=True,
                            ),
                            IO.Combo.Input(
                                "input_color_space",
                                options=["sRGB", "HDR", "HDR PQ"],
                                default="sRGB",
                                advanced=True,
                            ),
                            IO.Int.Input(
                                "crf",
                                default=18,
                                min=1,
                                max=63,
                                advanced=True,
                            ),
                            IO.DynamicCombo.Input(
                                "save_mode",
                                display_name="save mode",
                                options=[
                                    IO.DynamicCombo.Option("still images", []),
                                    IO.DynamicCombo.Option("animated", [
                                        IO.Float.Input(
                                            "fps",
                                            default=6.0,
                                            min=0.01,
                                            max=1000.0,
                                            step=0.01,
                                        ),
                                        IO.Int.Input(
                                            "loop_count",
                                            default=0,
                                            min=0,
                                            max=1000,
                                            advanced=True,
                                        ),
                                    ]),
                                ],
                            ),
                        ]),
                    ],
                    tooltip="The file format in which to save the image.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            outputs=[IO.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(
        cls,
        filename_prefix,
        counter_position,
        counter_padding,
        format,
        image_1,
        suffix_1="_albedo",
        image_2=None, suffix_2="_normal",
        image_3=None, suffix_3="_height",
        image_4=None, suffix_4="_roughness",
        image_5=None, suffix_5="_ao",
    ) -> IO.NodeOutput:
        file_format = format["format"]
        bit_depth = format["bit_depth"]
        colorspace = format.get("input_color_space", "sRGB")

        entries = []
        for img, suf in [
            (image_1, suffix_1),
            (image_2, suffix_2),
            (image_3, suffix_3),
            (image_4, suffix_4),
            (image_5, suffix_5),
        ]:
            if img is not None:
                entries.append((suf if suf is not None else "", img))

        if not entries:
            return IO.NodeOutput(torch.zeros(1, 1, 1, 3), ui={"images": []})

        # ------------------------------------------------------------------
        # Resolve tokens (%date:...) e subpasta via API nativa, mas ignora o
        # `counter` que ela retorna — pois ele só entende o formato do nó
        # nativo (<prefix>_<counter>.<ext>), não o formato deste nó.
        # ------------------------------------------------------------------
        first_image = entries[0][1][0]
        full_output_folder, filename, _, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
                first_image.shape[1],
                first_image.shape[0],
            )
        )

        # Contador próprio, baseado na posição configurada.
        counter = _find_max_counter(
            full_output_folder,
            filename,
            counter_padding,
            counter_position,
            file_format,
        )

        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo
        write_metadata = not args.disable_metadata

        results = []

        if file_format == "avif":
            metadata = None
            if write_metadata:
                metadata = {}
                if prompt is not None:
                    metadata["prompt"] = prompt
                if extra_pnginfo:
                    metadata.update(extra_pnginfo)

            save_mode = format["save_mode"]
            animated = save_mode["save_mode"] == "animated"

            if animated:
                for suffix, image_batch in entries:
                    name = _build_filename(
                        filename, counter, counter_padding,
                        suffix, counter_position
                    )
                    file = f"{name}.avif"
                    _save_avif(
                        image_batch,
                        os.path.join(full_output_folder, file),
                        bit_depth,
                        colorspace,
                        format["crf"],
                        fps=save_mode.get("fps", 1.0),
                        loop_count=save_mode.get("loop_count"),
                        metadata=metadata,
                    )
                    results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "output",
                    })
                counter += 1
            else:
                max_batch = max(batch.shape[0] for _, batch in entries)
                for batch_idx in range(max_batch):
                    for suffix, image_batch in entries:
                        if batch_idx >= image_batch.shape[0]:
                            continue
                        image = image_batch[batch_idx]
                        name = _build_filename(
                            filename, counter, counter_padding,
                            suffix, counter_position
                        )
                        file = f"{name}.avif"
                        _save_avif(
                            image.unsqueeze(0),
                            os.path.join(full_output_folder, file),
                            bit_depth,
                            colorspace,
                            format["crf"],
                            fps=1.0,
                            loop_count=None,
                            metadata=metadata,
                        )
                        results.append({
                            "filename": file,
                            "subfolder": subfolder,
                            "type": "output",
                        })
                    counter += 1

        else:
            max_batch = max(batch.shape[0] for _, batch in entries)
            for batch_idx in range(max_batch):
                for suffix, image_batch in entries:
                    if batch_idx >= image_batch.shape[0]:
                        continue
                    image = image_batch[batch_idx]

                    encoded = _encode_image(image, file_format, bit_depth, colorspace)

                    if write_metadata:
                        if file_format == "png":
                            encoded = inject_png_metadata(encoded, prompt, extra_pnginfo)
                        elif file_format == "exr":
                            encoded = inject_exr_metadata(
                                encoded, prompt, extra_pnginfo, colorspace
                            )

                    name = _build_filename(
                        filename, counter, counter_padding,
                        suffix, counter_position
                    )
                    file = f"{name}.{file_format}"
                    full_path = os.path.join(full_output_folder, file)
                    with open(full_path, "wb") as f:
                        f.write(encoded)

                    results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "output",
                    })
                counter += 1

        preview = entries[0][1]
        return IO.NodeOutput(preview, ui={"images": results})


class SavePBRTextureSetExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SavePBRTextureSet]


async def comfy_entrypoint() -> SavePBRTextureSetExtension:
    return SavePBRTextureSetExtension()


NODE_CLASS_MAPPINGS = {"SavePBRTextureSet": SavePBRTextureSet}
NODE_DISPLAY_NAME_MAPPINGS = {"SavePBRTextureSet": "Save PBR Texture Set"}