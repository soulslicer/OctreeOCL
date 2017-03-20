file(REMOVE_RECURSE
  "liboctree_ocl.pdb"
  "liboctree_ocl.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang)
  include(CMakeFiles/octree_ocl.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
