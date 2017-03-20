#include <iostream>
#include <cstdlib>
#include <time.h>

#include "Octree.hpp"



//#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <algorithm>

#include "OctreeOCL.hpp"

#include <typeinfo>

/** Example 4: Searching radius neighbors with default access by public x,y,z variables.
 *
 * \author behley
 */

#include <fstream>
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

template <typename PointT, typename ContainerT>
void readPoints(const std::string& filename, ContainerT& points)
{
  std::ifstream in(filename.c_str());
  std::string line;
  boost::char_separator<char> sep(" ");
  // read point cloud from "freiburg format"
  while (!in.eof())
  {
    std::getline(in, line);
    in.peek();

    boost::tokenizer<boost::char_separator<char> > tokenizer(line, sep);
    std::vector<std::string> tokens(tokenizer.begin(), tokenizer.end());

    if (tokens.size() != 6) continue;
    float x = boost::lexical_cast<float>(tokens[3]);
    float y = boost::lexical_cast<float>(tokens[4]);
    float z = boost::lexical_cast<float>(tokens[5]);

    points.push_back(PointT(x, y, z));
  }

  in.close();
}

bool twoVectorsEqual(std::vector<int>& a, std::vector<int>& b){
    if(a.size() != b.size()) return false;
    std::sort(a.begin(), a.end(), std::greater<int>());
    std::sort(b.begin(), b.end(), std::greater<int>());
    for(int i=0; i<a.size(); i++){
        if(a[i] != b[i]) return false;
    }
    return true;
}

bool twoVectorVectorsEqual(std::vector<std::vector<int>>& a, std::vector<std::vector<int>>& b){
    if(a.size() != b.size()) return false;
    for(int i=0; i<a.size(); i++){
        if(!twoVectorsEqual(a[i], b[i])) return false;
    }
    return true;
}


using namespace std;

typedef pcl::PointXYZ PointT;



int main(int argc, char** argv)
{    
    // Read filename
    if (argc < 2)
    {
        std::cerr << "filename of point cloud missing." << std::endl;
        return -1;
    }
    std::string filename = argv[1];

    // Load points
    std::vector<PointT> points;
    readPoints<PointT>(filename, points);
    std::cout << "Read " << points.size() << " points." << std::endl;
    if (points.size() == 0)
    {
        std::cerr << "Empty point cloud." << std::endl;
        return -1;
    }

    // Increase Points
//    std::vector<PointT> pointsTemp;
//    for(int i=0; i<points.size(); i++){
//        pointsTemp.push_back(points[i]);
//        pointsTemp.push_back(pcl::PointXYZ(points[i].x+0.01, points[i].y-0.01, points[i].z+0.01));
//        pointsTemp.push_back(pcl::PointXYZ(points[i].x+0.02, points[i].y-0.02, points[i].z+0.02));
//        pointsTemp.push_back(pcl::PointXYZ(points[i].x+0.03, points[i].y-0.03, points[i].z+0.03));
//        pointsTemp.push_back(pcl::PointXYZ(points[i].x+0.04, points[i].y-0.04, points[i].z+0.04));
//        pointsTemp.push_back(pcl::PointXYZ(points[i].x+0.05, points[i].y-0.05, points[i].z+0.05));
//        pointsTemp.push_back(pcl::PointXYZ(points[i].x+0.06, points[i].y-0.06, points[i].z+0.06));
//    }
//    points = pointsTemp;
//    std::random_shuffle(points.begin(), points.end());

    // Load PCL version of points
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPoints (new pcl::PointCloud<pcl::PointXYZ>);
    for(int i=0; i<points.size(); i++){
        pclPoints->push_back(pcl::PointXYZ(points[i].x, points[i].y, points[i].z));
    }

    int64_t begin, end;

    // Initializing the Octree with points from point cloud.
    OctreeGPU<PointT> octree;
    unibn::OctreeParams params;

    // Initialize
    begin = clock();
    octree.initialize(points);
    cout << "Init Time: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

    begin = clock();
    octree.linearizeTree();
    cout << "Lin Time: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

    begin = clock();
    octree.uploadTreeToGPU();
    cout << "GPU Upload: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (pclPoints);

    // Setup queries
    std::vector<pcl::PointXYZ> queries;
    for(int i=0; i<points.size(); i++){
        PointT testPoint = pclPoints->at(i);
        testPoint.x+=0.02;
        testPoint.z+=0.05;
        testPoint.y-=0.01;
        queries.push_back(testPoint);
    }
    std::random_shuffle(queries.begin(), queries.end());
    cout << queries.size() << endl;

    bool test_type = true;

    if(test_type){

        // PCL Test
        begin = clock();
        std::vector<std::vector<int>> pclResults(queries.size());
        for(int i=0; i<queries.size(); i++){
            std::vector<int> treeIndices;
            std::vector<float> treeDists;
            tree->radiusSearch(queries[i], 0.5, treeIndices, treeDists);
            pclResults[i] = treeIndices;
        }
        cout << "PCL Speed: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

        // Default Test
        begin = clock();
        std::vector<std::vector<int>> defaultResults(queries.size());
        for(int i=0; i<queries.size(); i++){
            std::vector<uint32_t> resultsDefault;
            octree.radiusNeighbors<unibn::L2Distance<PointT>>(queries[i], 0.5, resultsDefault);
            std::vector<int> resultsDefaultConv;
            for(uint32_t m : resultsDefault) resultsDefaultConv.push_back((int)m);
            defaultResults[i] = resultsDefaultConv;
        }
        cout << "Default Speed: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

        // Non Recursive Test
        begin = clock();
        std::vector<std::vector<int>> nonRecursiveResults(queries.size());
        for(int i=0; i<queries.size(); i++){
            std::vector<int> resultsNonRecursive = octree.radiusNeighboursExt(queries[i], 0.5);
            nonRecursiveResults[i] = resultsNonRecursive;
        }
        cout << "NR Speed: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

        if(!twoVectorVectorsEqual(pclResults, defaultResults)) cout << "FUCK" << endl;

        if(!twoVectorVectorsEqual(nonRecursiveResults, defaultResults)) cout << "FUCK" << endl;

//        for(int i=0; i<queries.size(); i++){

//            std::vector<int> treeIndices;
//            std::vector<float> treeDists;
//            tree->radiusSearch(queries[i], 0.5, treeIndices, treeDists);

//            std::vector<uint32_t> resultsDefault;
//            octree.radiusNeighbors<unibn::L2Distance<PointT>>(queries[i], 0.5, resultsDefault);
//            std::vector<int> resultsDefaultConv;
//            for(uint32_t m : resultsDefault) resultsDefaultConv.push_back((int)m);

//            std::vector<int> resultsNonRecursive = octree.radiusNeighboursExt(queries[i], 0.5);

//            if(!twoVectorsEqual(resultsNonRecursive, resultsDefaultConv)) cout << "FUCK" << endl;

//            if(!twoVectorsEqual(treeIndices, resultsDefaultConv)) cout << "FUCK" << endl;

//        }



    }
    else
    {
        // PCL Test
        begin = clock();
        std::vector<int> indicesPCL(queries.size());
        for(int i=0; i<queries.size(); i++){
            std::vector<int> treeIndices (1);
            std::vector<float> treeDists (1);
            tree->nearestKSearch(queries[i], 1, treeIndices, treeDists);
            indicesPCL[i] = treeIndices[0];
        }
        cout << "PCL Speed: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

        // GPU Test
        begin = clock();
        std::vector<int> indicesGPU = octree.findNeighboursGPU(queries);
        cout << "GPU Speed: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

        // CPU Test
        begin = clock();
        std::vector<int> indicesCPU(queries.size());
        for(int i=0; i<queries.size(); i++){
            indicesCPU[i] = octree.findNeighbor(queries[i]);
        }
        cout << "CPU Speed: " << ((double)(clock() - begin) / CLOCKS_PER_SEC) << endl;

        // Confirm
        for(int i=0; i<queries.size(); i++){
            if(indicesCPU[i] != indicesGPU[i]) cout << "SHIT CPU/GPU MISMATCH " << i << endl;
            if(indicesPCL[i] != indicesGPU[i]) cout << "SHIT PCL/GPU MISMATCH " << i << endl;
        }
    }

    return 0;
}


//// radiusNeighbors returns indexes to neighboring points.
//std::vector<uint32_t> results;
//const Point3f& q = points[0];
//octree.radiusNeighbors<unibn::L2Distance<Point3f> >(q, 0.2f, results);
//std::cout << results.size() << " radius neighbors (r = 0.2m) found for (" << q.x << ", " << q.y << "," << q.z << ")"
//          << std::endl;
//for (uint32_t i = 0; i < results.size(); ++i)
//{
//  const Point3f& p = points[results[i]];
//  std::cout << "  " << results[i] << ": (" << p.x << ", " << p.y << ", " << p.z << ") => "
//            << std::sqrt(unibn::L2Distance<Point3f>::compute(p, q)) << std::endl;
//}

//// performing queries for each point in point cloud
//begin = clock();
//cout << "SIZE: " << points.size() << endl;
////#pragma omp parallel for shared(octree, points)
//for (uint32_t i = 0; i < points.size(); ++i)
//{
//  //cout << i << endl;
//  std::vector<uint32_t> result;
//  //octree.findNeighbor<unibn::L2Distance<Point3f> >(points[i]);
//  octree.radiusNeighbors<unibn::L2Distance<Point3f> >(points[i], 0.5f, result);
//}
//end = clock();
//double search_time = ((double)(end - begin) / CLOCKS_PER_SEC);
//std::cout << "Searching for all radius neighbors (r = 0.5m) took " << search_time << " seconds." << std::endl;

//octree.clear();

///* PCL TEST */
//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//tree->setInputCloud (pclPoints);

//begin = clock();
////#pragma omp parallel for shared(pclPoints, tree)
//for(int i=0; i<pclPoints->size(); i++){
//    pcl::PointXYZ& point = pclPoints->at(i);
//    std::vector<int> treeIndices (1);
//    std::vector<float> treeDists (1);
//    //tree->nearestKSearch(point, 1, treeIndices, treeDists);
//    tree->radiusSearch(point, 0.5, treeIndices, treeDists);
//}
//end = clock();
//search_time = ((double)(end - begin) / CLOCKS_PER_SEC);
//std::cout << "Searching for all radius neighbors (r = 0.5m) took " << search_time << " seconds." << std::endl;
////  endTime = boost::posix_time::microsec_clock::local_time();
////  cout << "CPU Time: " << timeDifference(startTime, endTime) << endl;
