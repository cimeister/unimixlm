#pragma once

#include <string>
#include <vector>
#include "core/math/vector3f.h"

namespace stl {
    
  struct triangle {
    Vector3f normal;
    Vector3f v1;
    Vector3f v2;
    Vector3f v3;
    triangle(Vector3f normalp, Vector3f v1p, Vector3f v2p, Vector3f v3p) :
      normal(normalp), v1(v1p), v2(v2p), v3(v3p) {}
  };

  std::ostream& operator<<(std::ostream& out, const triangle& t);

  struct stl_data {
    std::string name;
    std::vector<triangle> triangles;

    stl_data(std::string namep) : name(namep) {}
  };

  stl_data parse_stl(const std::string& stl_path);

}