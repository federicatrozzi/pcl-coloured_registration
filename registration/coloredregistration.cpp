/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014, Federica Trozzi (federicatrozzi@hotmail.it)
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/cloud_iterator.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/impl/sift_keypoint.hpp>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/vfh.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
// STL
#include <iostream>
#include <pcl/registration/ia_ransac.h>
// PCL
#include <pcl/common/io.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointTC;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointTC> PointCloudColor;
using pcl::visualization::PointCloudColorHandlerCustom;
//typedef typename pcl::registration::CorrespondenceEstimationBase<PointT, PointT> CorrespondenceEstimation;

struct PCD
{
  PointCloudColor::Ptr cloudD;
  std::string f_name;

  PCD() : cloudD (new PointCloudColor) {};
};

//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
 
/*
boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);  
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
  viewer->addCoordinateSystem (0.2);
  viewer->initCameraParameters ();

  return (viewer);
}
*/

pcl::PointCloud<pcl::Normal>::Ptr estimateSurfaceNormals (const pcl::PointCloud<pcl::PointXYZ>::Ptr & input, float radius)
{
	  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
	  normal_estimation.setSearchMethod (pcl::search::Search<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
	  normal_estimation.setRadiusSearch (radius);
	  normal_estimation.setInputCloud (input);
	  pcl::PointCloud< pcl::Normal >::Ptr normals (new pcl::PointCloud< pcl::Normal >);
	  normal_estimation.compute (*normals);

	  return (normals);
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeLocalDescriptors (const pcl::PointCloud< pcl::Normal >::Ptr & normals, const pcl::PointCloud<pcl::PointXYZ>::Ptr & keypoints, float feature_radius)
{

	  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
	  fpfh_estimation.setSearchMethod (pcl::search::Search<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
	  fpfh_estimation.setRadiusSearch (feature_radius);
	  //fpfh_estimation.setSearchSurface (points);  
	  fpfh_estimation.setInputNormals (normals);
	  fpfh_estimation.setInputCloud (keypoints);
	  pcl::PointCloud<pcl::FPFHSignature33>::Ptr local_descriptors (new pcl::PointCloud<pcl::FPFHSignature33>);
	  fpfh_estimation.compute (*local_descriptors);
	  cout << " Local Descriptor : " << endl;
	  cout << local_descriptors->getMatrixXfMap()<< endl;
	  return (local_descriptors);
}

Eigen::Matrix4f computeInitialAlignment (const pcl::PointCloud<pcl::PointXYZ>::Ptr & source_points, const pcl::PointCloud<pcl::FPFHSignature33>::Ptr & source_descriptors,
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr & target_points, const pcl::PointCloud<pcl::FPFHSignature33>::Ptr & target_descriptors,
                         float min_sample_distance, float max_correspondence_distance, int nr_iterations)
{

	  pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
	  sac_ia.setMinSampleDistance (min_sample_distance);
	  sac_ia.setMaxCorrespondenceDistance (max_correspondence_distance);
	  sac_ia.setMaximumIterations (nr_iterations);
  
	  sac_ia.setInputCloud (source_points);
	  sac_ia.setSourceFeatures (source_descriptors);

	  sac_ia.setInputTarget (target_points);
	  sac_ia.setTargetFeatures (target_descriptors);
	  
	  pcl::PointCloud<pcl::PointXYZ> registration_output;
	  sac_ia.align (registration_output);
	  

	  return (sac_ia.getFinalTransformation ());
}

Eigen::Matrix4f
refineAlignment ( const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_points, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_points, 
                  const Eigen::Matrix4f &initial_alignment, float max_correspondence_distance,
                  float outlier_rejection_threshold, float transformation_epsilon, float max_iterations )
{

	  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp ;
	  icp.setMaxCorrespondenceDistance (max_correspondence_distance);
	  icp.setRANSACOutlierRejectionThreshold (outlier_rejection_threshold);
	  icp.setTransformationEpsilon (transformation_epsilon);
	  icp.setMaximumIterations (max_iterations);

	  pcl::PointCloud<pcl::PointXYZ>::Ptr source_points_transformed (new pcl::PointCloud<pcl::PointXYZ>);
	  pcl::transformPointCloud (*source_points, *source_points_transformed, initial_alignment);

	  icp.setInputCloud (source_points_transformed);
	  icp.setInputTarget (target_points);

	  //pcl::PointCloud<pcl::PointXYZ> registration_output;
	  //icp.align (registration_output);
	       
	  return (icp.getFinalTransformation() *initial_alignment );
}
	  

Eigen::Matrix4f
ICP ( const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_points, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_points, float max_correspondence_distance,
                  float outlier_rejection_threshold, float transformation_epsilon, float max_iterations )
{

	  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp ;
	  icp.setMaxCorrespondenceDistance (max_correspondence_distance);
	  //icp.setRANSACIterations(0);//RANSAC needs the below parameter to work unless it is zero, default RANSAC properties are 1000 iterations and 0.05 distance threshold
	  icp.setRANSACOutlierRejectionThreshold (outlier_rejection_threshold);
	  icp.setTransformationEpsilon (transformation_epsilon);
	  icp.setMaximumIterations (max_iterations);
	  //pcl::PointCloud<pcl::PointXYZ>::Ptr source_points_transformed (new pcl::PointCloud<pcl::PointXYZ>);
	  //pcl::transformPointCloud (*source_points, *source_points_transformed, initial_alignment);
	  icp.setInputSource (source_points);
	  icp.setInputTarget (target_points);
	  pcl::PointCloud<pcl::PointXYZ> registration_output;
	  icp.align (registration_output);
	 
       
	  return (icp.getFinalTransformation() );
}
	





int main (int argc, char** argv)
{
   
	 const float min_scale = 0.01f; //0.05 0.01f
    const int n_octaves = 3; //6 3 
    const int n_scales_per_octave = 4; //10 4 
    const float min_contrast = 0.1; //0.5f 0.001f

    //const float radiusN = 0.03;// 0.01;

	//const float Feat_radius =0.05;// 0.03;

	//const float min_samp_dist =0.05f;// 0.1;
	const float max_corrisp_dist =0.05 ;// 0.05 0.1; 5 centimetri
	//const int num_it=500;//2;

	const float outl_rej_thresh =0.3;//0.5
	const float transf_eps=2e-4; // sarebbe 9.55  //10e-4 //5e-4
	const int max_it=500;
	    
	Eigen::Affine3f aff;

	float x,y,z,roll,pitch,yaw;

    Eigen::Matrix4f init_align, fin_align, fin_align_true;

	Eigen::Matrix4f prev = Eigen::Matrix4f::Identity ();

    std::vector<PCD, Eigen::aligned_allocator<PCD> > data;

    //----------------------Allocazione delle point cloud in memoria------------------
	for (int i = 1; i < argc; i++)
    {
       PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloudD);
      //remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*m.cloudD,*m.cloudD, indices);
	  
      data.push_back (m);
      
      }
      //----------------------fine allocazione----------------------------------------


    //PointCloudColor::Ptr cloud_src, cloud_tgt;
	PointCloudColor::Ptr src, tgt ;
	//PointCloudColor::Ptr tgt (new PointCloudColor);
	PointCloudColor::Ptr cloud_result (new PointCloudColor);
	PointCloudColor::Ptr cloud_result2 (new PointCloudColor);
	PointCloudColor::Ptr output (new PointCloudColor);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_result); ok
	
	//viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
	//viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	//viewer->addCoordinateSystem (0.2);
	viewer->initCameraParameters ();

	for (int i = 0; i < data.size()-1; ++i)
	{ 
    
	 //pcl::PointCloud< pcl::Normal >::Ptr normals1 (new pcl::PointCloud< pcl::Normal >);
	 //pcl::PointCloud< pcl::Normal >::Ptr normals2 (new pcl::PointCloud< pcl::Normal >);


	//cloud_src = data[i].cloudD;
    //cloud_tgt = data[i+1].cloudD;

	//PointCloudColor::Ptr src (new PointCloudColor);
    //PointCloudColor::Ptr tgt (new PointCloudColor);
/*
	pcl::VoxelGrid<pcl::PointXYZRGB> grid;
	grid.setLeafSize (0.01f, 0.01f, 0.01f);
 
	grid.setInputCloud (cloud_src);
	grid.filter (*src);

	grid.setInputCloud (cloud_tgt);
	grid.filter (*tgt);
  */
	

	if (i!= 0)
	{
		
    src = output;
	}

	else {

		src = data[i].cloudD;
	}

	tgt = data[i+1].cloudD;

     //-----------------------Estimate the sift interest points using Intensity values from RGB values----------------
    pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result, result2;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(src);
    sift.compute(result);
    sift.setInputCloud(tgt);
    sift.compute(result2);
	

     // Copying the pointwithscale to pointxyz so as visualize the cloud
     pcl::PointCloud<pcl::PointXYZ>::Ptr keypointsxyz (new pcl::PointCloud<pcl::PointXYZ>);
     copyPointCloud(result, *keypointsxyz);

     pcl::PointCloud<pcl::PointXYZ>::Ptr keypointsxyz2 (new pcl::PointCloud<pcl::PointXYZ>);
     copyPointCloud(result2, *keypointsxyz2);

     std::cout<< "keipoint PRIMA point cloud"<<std::endl;

         for (int i=0; i<keypointsxyz->points.size (); ++i)
       {
		std::cerr << "    " << keypointsxyz->points[i].x<< " " << keypointsxyz->points[i].y << " " << keypointsxyz->points[i].z << std::endl;
       }

	 std::cout<< "keipoint SECONDA point cloud"<<std::endl;

         for (int i=0; i<keypointsxyz2->points.size (); ++i)
       {
		std::cerr << "    " << keypointsxyz2->points[i].x<< " " << keypointsxyz2->points[i].y << " " << keypointsxyz2->points[i].z << std::endl;
		 }
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (keypointsxyz, 0, 255, 0);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler2 (keypointsxyz2, 255, 0, 0);
    
	//------------------ESTIMATE NORMALS--------------------//

	//normals1=estimateSurfaceNormals (keypointsxyz, radiusN);


	//normals2=estimateSurfaceNormals (keypointsxyz2, radiusN);

	//-----------------------END-----------------------------//



    //--------------------COMPUTE DESCRIPTORS--------------------//
		/* pcl::PointCloud<pcl::FPFHSignature33>::Ptr local_descriptors1 (new pcl::PointCloud<pcl::FPFHSignature33>);

		 pcl::PointCloud<pcl::FPFHSignature33>::Ptr local_descriptors2 (new pcl::PointCloud<pcl::FPFHSignature33>);

		 local_descriptors1=computeLocalDescriptors (normals1, keypointsxyz , Feat_radius);

		 local_descriptors2=computeLocalDescriptors (normals2, keypointsxyz2 , Feat_radius);
*/
	//------------------------END-------------------------------//


   //-------------------------COMPUTE INITIAL TRANSFORMATION-------------//
		//init_align = computeInitialAlignment (keypointsxyz, local_descriptors1, keypointsxyz2, local_descriptors2, min_samp_dist ,  max_corrisp_dist , num_it);


		//std::cout<<init_align<<std::endl;


   //------------------------REFINE ALIGNMENT---------------------//

		
	//fin_align=refineAlignment (keypointsxyz, keypointsxyz2 ,init_align, max_corrisp_dist, outl_rej_thresh ,transf_eps, max_it );

	fin_align = ICP (keypointsxyz, keypointsxyz2, max_corrisp_dist, outl_rej_thresh ,transf_eps, max_it );
	
	aff.matrix()=fin_align;

	
	std::cout<<fin_align<<std::endl;

	//std::cout<<prev<<std::endl;
	
	pcl::getTranslationAndEulerAngles(aff,x,y,z,roll,pitch,yaw);

	std::cout<<x<<"  "<<roll<<"  "<<std::endl;

	std::cout<<y<<"  "<<pitch<<"  "<<std::endl;

	std::cout<<z<<"  "<<yaw<<"  "<<std::endl;

	std::cout<<"  "<<std::endl;
	//PointCloudColor::Ptr output (new PointCloudColor);ok2
	PointCloud::Ptr output2 (new PointCloud);

  //fin_align=fin_align*prev;
  //fin_align_true=fin_align.inverse();

  pcl::transformPointCloud (*src, *output, fin_align);
  pcl::transformPointCloud (*keypointsxyz, *output2, fin_align);
  //pcl::transformPointCloud (*tgt, *output, fin_align_true);
  //std::cout<<fin_align_true<<std::endl;

  prev=fin_align;

  *cloud_result += *output;
  *cloud_result += *src;


  //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler2 (output2, 255, 0, 0);ok
  std::cout<<data.size()<<std::endl;
  //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(src);ok
  //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(output);ok
  //viewer->addPointCloud<pcl::PointXYZRGB> (output, rgb, argv[i]);ok
  //viewer->addPointCloud<pcl::PointXYZRGB> (src, rgb2,argv[i+1]);ok
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_result);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud_result, rgb,argv[i]);
  //viewer->addPointCloud(keypointsxyz, keypoints_color_handler, "keypoints");ok
  //viewer->addPointCloud(keypointsxyz2, keypoints_color_handler2, "keypoints2");ok
  //viewer->addPointCloud(output2, keypoints_color_handler2, "keypoints2");
  viewer->addCoordinateSystem (0.2,aff);
  viewer->updatePointCloudPose(argv[i],aff);
  viewer->updatePointCloud(cloud_result, argv[i]);
  viewer->spinOnce();
  
  }

  //viewer = rgbVis(cloud_result);
  
  viewer->spin();

}
  
  
/*

pcl::PointCloud<pcl::FPFHSignature33>::Ptr
computeLocalDescriptors (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & points(input cloud), const pcl::PointCloud< pcl::Normal >::Ptr & normals, 
                         const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & keypoints, float feature_radius)
{
	  pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
	  fpfh_estimation.setSearchMethod (pcl::search::Search<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
	  fpfh_estimation.setRadiusSearch (feature_radius);
	  fpfh_estimation.setSearchSurface (points);  
	  fpfh_estimation.setInputNormals (normals);
	  fpfh_estimation.setInputCloud (keypoints);
	  pcl::PointCloud<pcl::FPFHSignature33>::Ptr local_descriptors (new pcl::PointCloud<pcl::FPFHSignature33>);
	  fpfh_estimation.compute (*local_descriptors);
	  cout << " Local Descriptor : " << endl;
	  cout << local_descriptors->getMatrixXfMap()<< endl;
	  return (local_descriptors);
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr
detectKeypoints (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & points, const pcl::PointCloud< pcl::Normal >::Ptr & normals,
                 float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast)
{
	  pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;
	  sift_detect.setSearchMethod (pcl::search::Search<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
	  sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
	  sift_detect.setMinimumContrast (min_contrast);
	  sift_detect.setInputCloud (points);
	  pcl::PointCloud<pcl::PointWithScale> keypoints_temp;
	  sift_detect.compute (keypoints_temp);
	  pcl::PointCloud< pcl::PointXYZRGB >::Ptr keypoints (new pcl::PointCloud< pcl::PointXYZRGB > );
	  pcl::copyPointCloud (keypoints_temp, *keypoints);

	  return (keypoints);
}


pcl::PointCloud<pcl::Normal>::Ptr
estimateSurfaceNormals (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & input, float radius)
{
	  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
	  normal_estimation.setSearchMethod (pcl::search::Search<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
	  normal_estimation.setRadiusSearch (radius);
	  normal_estimation.setInputCloud (input);
	  pcl::PointCloud< pcl::Normal >::Ptr normals (new pcl::PointCloud< pcl::Normal >);
	  normal_estimation.compute (*normals);

	  return (normals);
}

*/

 

