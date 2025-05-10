#include "extensions/ess/components/ess_inference.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xce7c6985267a4ec7, 0xa073030e16e49f29, "ESS",
                         "Extension containing ESS related components",
                         "Isaac SDK", "2.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x1aa1eea914344afe, 0x97fddaaedb594120,
                    nvidia::isaac::ESSInference, nvidia::gxf::Codelet,
                    "ESS GXF Extension");

GXF_EXT_FACTORY_END()
