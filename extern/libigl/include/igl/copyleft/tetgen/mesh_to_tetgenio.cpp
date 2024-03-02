// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "mesh_to_tetgenio.h"

// IGL includes 
#include "../../matrix_to_list.h"

// STL includes
#include <cassert>

template <
  typename DerivedV, 
  typename DerivedF, 
  typename DerivedH, 
  typename DerivedVM, 
  typename DerivedFM, 
  typename DerivedR>
IGL_INLINE void igl::copyleft::tetgen::mesh_to_tetgenio(
  const Eigen::MatrixBase<DerivedV>& V,
  const Eigen::MatrixBase<DerivedF>& F,
  const Eigen::MatrixBase<DerivedH>& H,
  const Eigen::MatrixBase<DerivedVM>& VM,
  const Eigen::MatrixBase<DerivedFM>& FM,
  const Eigen::MatrixBase<DerivedR>& R,
  tetgenio & in)
{
  using namespace std;
  assert(V.cols() == 3 && "V should have 3 columns");
  assert((VM.size() == 0 || VM.size() == V.rows()) && "VM should be empty or #V by 1");
  assert((FM.size() == 0 || FM.size() == F.rows()) && "FM should be empty or #F by 1");
  in.firstnumber = 0;
  in.numberofpoints = V.rows();
  in.pointlist = new REAL[in.numberofpoints * 3];
  if(VM.size())
  {
    in.pointmarkerlist = new int[VM.size()];
  }
  //loop over points
  for(int i = 0; i < V.rows(); i++)
  {
    in.pointlist[i*3+0] = V(i,0);
    in.pointlist[i*3+1] = V(i,1);
    in.pointlist[i*3+2] = V(i,2);    
    if(VM.size())
    {
      in.pointmarkerlist[i] = VM(i);
    }
  }
  in.numberoffacets = F.rows();
  in.facetlist = new tetgenio::facet[in.numberoffacets];
  in.facetmarkerlist = new int[in.numberoffacets];

  // loop over face
  for(int i = 0;i < F.rows(); i++)
  {
    in.facetmarkerlist[i] = FM.size() ? FM(i) : i;
    tetgenio::facet * f = &in.facetlist[i];
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
    f->numberofholes = 0;
    f->holelist = NULL;
    tetgenio::polygon * p = &f->polygonlist[0];
    p->numberofvertices = F.cols();
    p->vertexlist = new int[p->numberofvertices];
    // loop around face
    for(int j = 0;j < F.cols(); j++)
    {
      p->vertexlist[j] = F(i,j);
    }
  }
  
  in.numberofholes = H.rows(); 
  in.holelist = new REAL[3 * in.numberofholes];
  // loop over holes
  for(int holeID = 0; holeID < H.rows(); holeID++)
  {
    in.holelist[holeID * 3 + 0] = H(holeID,0); 
    in.holelist[holeID * 3 + 1] = H(holeID,1);
    in.holelist[holeID * 3 + 2] = H(holeID,2);
  }  

  in.numberofregions = R.rows();
  in.regionlist = new REAL[ 5 * in.numberofregions];
  // loop over regions
  for(int regionID = 0; regionID < R.rows(); regionID++)
  {
    in.regionlist[regionID * 5 + 0] = R(regionID,0); 
    in.regionlist[regionID * 5 + 1] = R(regionID,1);
    in.regionlist[regionID * 5 + 2] = R(regionID,2);
    in.regionlist[regionID * 5 + 3] = R(regionID,3);
    in.regionlist[regionID * 5 + 4] = R(regionID,4);
  }  

}


#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
// generated by autoexplicit.sh
template void igl::copyleft::tetgen::mesh_to_tetgenio<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>>(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const&, tetgenio&);
#endif
