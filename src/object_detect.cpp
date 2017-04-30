#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>

#include "arapaho.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>

ArapahoV2* p;
float thresh = 0.5;
std::string INPUT_DATA_FILE = ros::package::getPath("darknet_ros") + "/input.names";
std::string INPUT_CFG_FILE = ros::package::getPath("darknet_ros") + "/input.cfg";
std::string INPUT_WEIGHTS_FILE = ros::package::getPath("darknet_ros") + "/input.weights";

// static char INPUT_CFG_FILE[]     = "input.cfg";
// static char INPUT_WEIGHTS_FILE[] = "input.weights";

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    int numObjects = 0;
    bool ret = false;
    int expectedW = 0, expectedH = 0;
    box* boxes = 0;
    std::string* labels;

    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;

    if( !image.empty() ) 
    {

      int imageWidthPixels = image.size().width;
      int imageHeightPixels = image.size().height;
      DPRINTF("Image data = %p, w = %d, h = %d\n", image.data, imageWidthPixels, imageHeightPixels);

    // Remember the time
      auto detectionStartTime = std::chrono::system_clock::now();

      p->Detect(
        image,
        thresh,
        0.5,
        numObjects);

      std::chrono::duration<double> detectionTime = (std::chrono::system_clock::now() - detectionStartTime);

      printf("==> Detected [%d] objects in [%f] seconds\n", numObjects, detectionTime.count());

      if(numObjects > 0)
      {    
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
        for (objId = 0; objId < numObjects; objId++)
        {
          leftTopX = 1 + imageWidthPixels*(boxes[objId].x - boxes[objId].w / 2);
          leftTopY = 1 + imageHeightPixels*(boxes[objId].y - boxes[objId].h / 2);
          rightBotX = 1 + imageWidthPixels*(boxes[objId].x + boxes[objId].w / 2);
          rightBotY = 1 + imageHeightPixels*(boxes[objId].y + boxes[objId].h / 2);
          DPRINTF("Box #%d: center {x,y}, box {w,h} = [%f, %f, %f, %f]\n", 
            objId, boxes[objId].x, boxes[objId].y, boxes[objId].w, boxes[objId].h);
            // Show image and overlay using OpenCV
          cv::rectangle(image,
            cvPoint(leftTopX, leftTopY),
            cvPoint(rightBotX, rightBotY),
            CV_RGB(255, 0, 0), 1, 8, 0);
            // Show labels
          if (labels[objId].c_str())
          {
            DPRINTF("Label:%s\n\n", labels[objId].c_str());
            putText(image, labels[objId].c_str(), cvPoint(leftTopX, leftTopY),
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

      }// If objects were detected

          cv::imshow("view", image);
          cv::waitKey(30);
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
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/camera/image_raw", 1, imageCallback);
  
  // Initialize darknet object using Arapaho API
  p = new ArapahoV2();
  if(!p)
  {
    return -1;
  }

    // TODO - read from arapaho.cfg    
  ArapahoV2Params ap;
  ap.datacfg = (char *)INPUT_DATA_FILE.c_str();
  ap.cfgfile = (char *)INPUT_CFG_FILE.c_str();
  ap.weightfile = (char *)INPUT_WEIGHTS_FILE.c_str();
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

  ros::spin();
  cv::destroyWindow("view");
  if(p) delete p;
  DPRINTF("Exiting...\n");
  return 0;
}