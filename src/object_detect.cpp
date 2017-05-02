#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "darknet_ros/DetectedObjects.h"
#include "darknet_ros/ObjectInfo.h"

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>

#include "arapaho.hpp"
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>

ArapahoV2* p;
float thresh;
ros::Publisher objPub_;
image_transport::Publisher imgPub_;
cv_bridge::CvImagePtr cv_ptr_;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    int numObjects = 0;
    bool ret = false;
    int expectedW = 0, expectedH = 0;
    box* boxes = 0;
    std::string* labels;
    //cv_ptr_  = cv_bridge::toCvShare(msg, "bgr8");
    cv_ptr_ = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8);

    if( cv_ptr_.get()) 
    {

      int imageWidthPixels = cv_ptr_->image.size().width;
      int imageHeightPixels = cv_ptr_->image.size().height;
      DPRINTF("Image data = %p, w = %d, h = %d\n", cv_ptr_->image.data, imageWidthPixels, imageHeightPixels);

    // Remember the time
      auto detectionStartTime = std::chrono::system_clock::now();
      
      p->Detect(
        cv_ptr_->image,
        thresh,
        0.5,
        numObjects);
  
      std::chrono::duration<double> detectionTime = (std::chrono::system_clock::now() - detectionStartTime);
      darknet_ros::DetectedObjects tObjMsg;

      tObjMsg.header = cv_ptr_->header;
      printf("==> Detected [%d] objects in [%f] seconds\n", numObjects, detectionTime.count());

      if(numObjects > 0)
      {    
        
        std::vector<darknet_ros::ObjectInfo> objectList;
        boxes = new box[numObjects];
        labels = new std::string[numObjects];
     
        // Get boxes and labels
        p->GetBoxes(
          boxes,
          labels,
          numObjects
          );

        int objId = 0;
        int leftTopX = 0, leftTopY = 0, rightBotX = 0,rightBotY = 0;
        darknet_ros::ObjectInfo newObj;
        for (objId = 0; objId < numObjects; objId++)
        {
          
          leftTopX = 1 + imageWidthPixels*(boxes[objId].x - boxes[objId].w / 2);
          leftTopY = 1 + imageHeightPixels*(boxes[objId].y - boxes[objId].h / 2);
          rightBotX = 1 + imageWidthPixels*(boxes[objId].x + boxes[objId].w / 2);
          rightBotY = 1 + imageHeightPixels*(boxes[objId].y + boxes[objId].h / 2);
          DPRINTF("Box #%d: center {x,y}, box {w,h} = [%f, %f, %f, %f]\n", 
            objId, boxes[objId].x, boxes[objId].y, boxes[objId].w, boxes[objId].h);
            // Show image and overlay using OpenCV
          cv::rectangle(cv_ptr_->image,
            cvPoint(leftTopX, leftTopY),
            cvPoint(rightBotX, rightBotY),
            CV_RGB(255, 0, 0), 1, 8, 0);
          newObj.tl_x = leftTopX < 0 ? 0 : leftTopX;
          newObj.tl_y = leftTopY < 0 ? 0 : leftTopY;
          newObj.width = rightBotX - newObj.tl_x;
          newObj.height = rightBotY - newObj.tl_y;
          newObj.type = labels[objId];
          objectList.push_back(newObj);
            // Show labels
          if (labels[objId].c_str())
          {
            DPRINTF("Label:%s\n\n", labels[objId].c_str());
            putText(cv_ptr_->image, labels[objId].c_str(), cvPoint(leftTopX, leftTopY),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
          }
        }

        if (boxes)
        {
          delete[] boxes;
          boxes = NULL;
        }
        if (labels)
        {
          delete[] labels;
          labels = NULL;
        }   

      tObjMsg.objects = objectList;
      }// If objects were detected
      objPub_.publish(tObjMsg);
      imgPub_.publish(cv_ptr_->toImageMsg());
      cv::imshow("view", cv_ptr_->image);
      cv::waitKey(30);
      cv_ptr_.reset();
    }
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  cv::namedWindow("view");
  cv::startWindowThread();

  objPub_ = nh.advertise<darknet_ros::DetectedObjects>( "/darknet_ros/detected_objects", 1);
  

  std::string ros_path = ros::package::getPath("darknet_ros");
  
  ros::NodeHandle priNh( "~" );
  std::string yoloWeightsFile;
  std::string yoloConfigFile;
  std::string yoloDataFile;
  
  priNh.param<std::string>( "weights_file", yoloWeightsFile, ros_path + "/input.weights" );
  priNh.param<std::string>( "cfg_file", yoloConfigFile, ros_path+"/input.cfg" );
  priNh.param<std::string>( "data_file", yoloDataFile, ros_path+"/input.name" );
  priNh.param( "thresh", thresh, 0.2f );
/*
  std::string INPUT_DATA_FILE = ros_path + "/" + INPUT_DATA_FILE;
  std::string INPUT_WEIGHTS_FILE = ros_path + "/" + INPUT_WEIGHTS_FILE;
  std::string INPUT_CFG_FILE = ros_path + "/" + INPUT_CFG_FILE;
  thresh = std::stof(threshold);
*/
  // Initialize darknet object using Arapaho API
  p = new ArapahoV2();
  if(!p)
  {
    return -1;
  }

    // TODO - read from arapaho.cfg    
  ArapahoV2Params ap;
  ap.datacfg = (char *)yoloDataFile.c_str();//INPUT_DATA_FILE.c_str();
  ap.cfgfile = (char *)yoloConfigFile.c_str();//INPUT_CFG_FILE.c_str();
  ap.weightfile = (char *)yoloWeightsFile.c_str();//INPUT_WEIGHTS_FILE.c_str();
  ap.nms = 0.4;
  ap.maxClasses = 20;
  int expectedW = 0, expectedH = 0;
    // Always setup before detect
  bool ret = p->Setup(ap, expectedW, expectedH);
  if(false == ret)
  {
    EPRINTF("Setup failed!\n");
    if(p) delete p;
    p = 0;
    return -1;
  }

  image_transport::ImageTransport it(nh);  

  imgPub_ = it.advertise( "/darknet_ros/image", 1);
  image_transport::Subscriber sub = it.subscribe("/camera/image_raw", 1, imageCallback);
  
  ros::spin();
  cv::destroyWindow("view");
  if(p) delete p;
  DPRINTF("Exiting...\n");
  return 0;
}
