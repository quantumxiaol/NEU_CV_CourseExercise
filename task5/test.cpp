// C++
#include <iostream>
#include <chrono>
#include <string>
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// Kinect DK
#include <k4a/k4a.h>
using namespace cv;
using namespace std;

//  宏
//  方便控制是否 std::cout  信息
#define DEBUG_std_cout 0
int main(int argc, char* argv[]) {
	//获取设备数，确定是否连接了设备
	const uint32_t device_count = k4a_device_get_installed_count();
	if (0 == device_count) {
		std::cout << "Error: no K4A devices found. " << std::endl;
		return -1;
	}
	else {
		std::cout << "Found  " << device_count << "  connected  devices.  " <<
			std::endl;

		//  超过 1 个设备，也输出错误信息。 
		if (1 != device_count)
		{
			std::cout << "Error: more than one K4A devices found. " << std::endl;
			return -1;
		}
		//  该示例代码仅限对 1 个设备操作
		else {
			std::cout << "Done: found 1 K4A device. " << std::endl;
		}
	}
	//  打开 (默认) 设备
	k4a_device_t device = NULL;
	k4a_result_t device_open_result = k4a_device_open(K4A_DEVICE_DEFAULT, &device);
	if (device_open_result != K4A_RESULT_SUCCEEDED) {
		std::cout << "Failed to open k4a device!" << std::endl;
		return 1;
	}
	std::cout << "Done: open device. " << std::endl;
	/*
	检索并保存 Azure Kinect  图像数据
	*/
	//  配置并启动设备
	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.camera_fps = K4A_FRAMES_PER_SECOND_30;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	config.color_resolution = K4A_COLOR_RESOLUTION_720P;
	config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	config.synchronized_images_only = true;// ensures that depth and color images are both available in the capture
	k4a_result_t device_start_result = k4a_device_start_cameras(device, &config);
	if (device_start_result != K4A_RESULT_SUCCEEDED) {
		std::cout << "Failed to start cameras!" << std::endl;
		k4a_device_close(device);
		return 1;
	}
	else {
		std::cout << "Done: start camera." << std::endl;
	}
	//查询传感器校准
	k4a_calibration_t sensor_calibration;
	k4a_result_t get_calibration_result = k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &sensor_calibration);
	if (get_calibration_result != K4A_RESULT_SUCCEEDED) {
		std::cout << "Get depth camera calibration failed!" << std::endl;
		return 1;
	}
	//  稳定化
	k4a_capture_t capture;
	int iAuto = 0;//用来稳定，类似自动曝光
	int iAutoError = 0;//  统计自动曝光的失败次数
	while (true) {
		k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &capture, K4A_WAIT_INFINITE);
		if (get_capture_result == K4A_RESULT_SUCCEEDED) {
			std::cout << iAuto << ". Capture several frames to give auto-exposure" << std::endl;
			//  跳过前 n  个 (成功的数据采集) 循环，用来稳定
			if (iAuto != 30) {
				iAuto++;
				continue;
			}
			else {
				std::cout << "Done: auto-exposure" << std::endl;
				break;//  跳出该循环，完成相机的稳定过程
			}
		}
		else {
			std::cout << iAutoError << ". K4A  WAIT  RESULT  TIMEOUT." <<
				std::endl;
			if (iAutoError != 30) {
				iAutoError++;
				continue;
			}
			else {
				std::cout << "Error: failed to give auto-exposure. " << std::endl; return -1;
			}
		}
	}
	std::cout << "-----------------------------------" << std::endl;
	std::cout << "----- Have Started Kinect DK. -----" << std::endl;
	std::cout << "-----------------------------------" << std::endl;
	//  从设备获取捕获
	k4a_image_t rgbImage;
	k4a_image_t depthImage;
	k4a_image_t irImage;
	cv::Mat cv_rgbImage_with_alpha;
	cv::Mat cv_rgbImage_no_alpha;
	cv::Mat cv_depth;
	cv::Mat cv_depth_8U;
	cv::Mat cv_irImage;
	cv::Mat cv_irImage_8U;
	cv::Mat cv_transformed_depth;
	cv::Mat cv_transformed_depth_8U;

	while (true)
	{

		if (k4a_device_get_capture(device, &capture, K4A_WAIT_INFINITE) == K4A_RESULT_SUCCEEDED) {
			//k4a_image_t transformed_depthImage = nullptr;
			// rgb
			// * Each pixel of BGRA32 data is four bytes. The first three bytes represent Blue, Green,
			// * and Red data. The fourth byte is the alpha channel and is unused in the Azure Kinect APIs.
			rgbImage = k4a_capture_get_color_image(capture);
			#if DEBUG_std_cout == 1
				std::cout << "[rgb] " << "\n"
				<< "format: " << rgbImage.get_format() << "\n"
				<< "device_timestamp:                      " << rgbImage.get_device_timestamp().count() << "\n"
				<< "system_timestamp:                      " << rgbImage.get_system_timestamp().count() << "\n"
				<< "height*width: " << rgbImage.get_height_pixels() << ", " << rgbImage.get_width_pixels()
				<< std::endl;
#endif
			// depth
			// * Each pixel of DEPTH16 data is two bytes of little endian unsigned depth data. The unit of the data is in
			// * millimeters from the origin of the camera.
			depthImage = k4a_capture_get_depth_image(capture);
			#if DEBUG_std_cout == 1
				std::cout << "[depth] " << "\n"
				<< "format: " << depthImage.get_format() << "\n"
				<< "device_timestamp:                      " << depthImage.get_device_timestamp().count() << "\n"
				<< "system_timestamp:                      " << depthImage.get_system_timestamp().count() << "\n"
				<< "height*width:  " << depthImage.get_height_pixels() << ",  " << depthImage.get_width_pixels()
				<< std::endl;
#endif
			//ir
			//  *  Each  pixel  of IR16  data  is  two  bytes  of little  endian unsigned depth data. The value of the data represents
			// * brightness.
			irImage = k4a_capture_get_ir_image(capture);
			#if DEBUG_std_cout == 1

				std::cout << "[ir] " << "\n"
				<< "format: " << irImage.get_format() << "\n"
				<< "device_timestamp:                      " << irImage.get_device_timestamp().count() << "\n"
				<< "system_timestamp:                      " << irImage.get_system_timestamp().count() << "\n"
				<< "height*width:  " << irImage.get_height_pixels() << ",  " << irImage.get_width_pixels()
				<< std::endl;
#endif
			//深度图和 RGB 图配准
			k4a_calibration_t k4aCalibration = sensor_calibration;//获取相机标定参数




			k4a_transformation_t k4aTransformation = k4a_transformation_create(&k4aCalibration);
			//RGB
			cv_rgbImage_with_alpha = cv::Mat(k4a_image_get_height_pixels(rgbImage), k4a_image_get_width_pixels(rgbImage), CV_8UC4, k4a_image_get_buffer(rgbImage));
			cv::cvtColor(cv_rgbImage_with_alpha, cv_rgbImage_no_alpha, cv::COLOR_BGRA2BGR);







			//depth
			cv_depth = cv::Mat(k4a_image_get_height_pixels(depthImage), k4a_image_get_width_pixels(depthImage), CV_16U, k4a_image_get_buffer(depthImage), k4a_image_get_stride_bytes(depthImage));
			cv::normalize(cv_depth, cv_depth_8U, 0, 256 * 256, NORM_MINMAX);
			cv_depth_8U.convertTo(cv_depth, CV_8U, 1);
			//IR
			cv_irImage = cv::Mat(k4a_image_get_height_pixels(irImage), k4a_image_get_width_pixels(irImage), CV_16U, k4a_image_get_buffer(irImage), k4a_image_get_stride_bytes(irImage));
			cv_irImage.convertTo(cv_irImage_8U, CV_8U, 1);
			// show image
			cv::imshow("color", cv_rgbImage_no_alpha);
			cv::imshow("depth", cv_depth_8U);
			cv::imshow("ir", cv_irImage_8U);
			double time_rgb = static_cast<double>(std::chrono::microseconds(k4a_image_get_device_timestamp_usec(rgbImage)).count());
			std::string filename_rgb = std::to_string(time_rgb / 1000000) + ".png";
			double time_d = static_cast<double>(std::chrono::microseconds(k4a_image_get_device_timestamp_usec(depthImage)).count());
			std::string filename_d = std::to_string(time_d / 1000000) + ".png";
			double time_ir = static_cast<double>(std::chrono::microseconds(k4a_image_get_device_timestamp_usec(irImage)).count());
			std::string filename_ir = std::to_string(time_ir / 1000000) + ".png";


			imwrite("E:\\c\\workinVS\\cv\\image\\rgb\\" + filename_rgb, cv_rgbImage_no_alpha);
			imwrite("E:\\c\\workinVS\\cv\\image\\depth\\" + filename_d, cv_depth_8U);
			imwrite("E:\\c\\workinVS\\cv\\image\\ir\\" + filename_ir, cv_irImage_8U);
			std::cout << "Acquiring!" << endl;

			cv_rgbImage_with_alpha.release();
			cv_rgbImage_no_alpha.release();
			cv_depth.release();
			cv_depth_8U.release();
			cv_irImage.release();
			cv_irImage_8U.release();
			cv_transformed_depth.release();
			cv_transformed_depth_8U.release();
			k4a_image_release(rgbImage);
			k4a_image_release(depthImage);
			k4a_image_release(irImage);


			if (cv::waitKey(1) == 10)
				break;
			k4a_capture_release(capture);
		}
		else {
			std::cout << "false: K4A_WAIT_RESULT_TIMEOUT." << std::endl;
		}
	}
	cv::destroyAllWindows();



	//  释放，关闭设备
	k4a_capture_release(capture);
	k4a_device_stop_cameras(device);
	k4a_device_close(device);
	return 0;
}