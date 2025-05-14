

#include <algorithm>
#include <filesystem>  // NOLINT(build/include_order)

#include "common/fixed_vector.hpp"
#include "common/span.hpp"
#include "gems/gxf_helpers/expected_macro_gxf.hpp"
#include "gems/hash/hash_file.hpp"

namespace nvidia {
namespace isaac {

namespace {

constexpr size_t kBlockSize = 8096;

}  // namespace

// helper to get the hash of a file
gxf::Expected<SHA256::String> hash_file(const char* path) {
  std::ifstream file;
  SHA256 hash;
  FixedVector<uint8_t, kBlockSize> buffer;

  size_t bytes_remaining = std::filesystem::file_size(path);

  file.open(path, std::ios_base::in | std::ios_base::binary);

  if (!file.good()) {
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  while (true) {
    size_t current_bytes = std::min(kBlockSize, bytes_remaining);

    file.read(reinterpret_cast<char*>(buffer.data()), current_bytes);

    if (!file.good()) {
      return nvidia::gxf::Unexpected{GXF_FAILURE};
    }

    RETURN_IF_ERROR(hash.hashData(Span<uint8_t>(buffer.data(), current_bytes)));

    bytes_remaining -= current_bytes;

    if (bytes_remaining == 0) {
      RETURN_IF_ERROR(hash.finalize());

      return hash.toString();
    }
  }
}

}  // namespace isaac
}  // namespace nvidia
