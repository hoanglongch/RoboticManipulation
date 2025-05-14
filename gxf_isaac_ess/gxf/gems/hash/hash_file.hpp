
#pragma once

#include "gems/hash/sha256.hpp"

namespace nvidia {
namespace isaac {

// helper to get the hash of a file
gxf::Expected<SHA256::String> hash_file(const char* path);

}  // namespace isaac
}  // namespace nvidia
