**双目视觉实验

实现利用 Azure Kinect DK 获取并保存图像

配置Azure Kinect。
安装SDK。
在github上下载Azure Kinect SDK，安装到C盘默认路径。
执行k4aviewer，配置好参数。

遇到了bug可以试着更换USB接口。

保存并获取图像。
在IDE中安装 Azure Kinect NuGet 包。

配置opencv，我在官网下载的是4.7.0。
配置环境变量，复制lib文件到C盘。


重启电脑使配置生效。
编写代码，调用摄像头获取图像。

**人体追踪

Azure Kinect 人体跟踪 SDK

进入 C:\Program Files\Azure Kinect Body Tracking SDK\tools\文件夹中，有一个 k4abt_simple_3d_viewer 可 执 行 文 件 ， 可 以 在 Powershell 中 运 行 。
命令是：./k4abt_simple_3d_viewer.exe CPU。

之前代码报出错误，k4a_device_get_capture()有问题。
这个结果表示你没有成功获取捕获对象，函数返回了K4A_WAIT_RESULT_FAILED。这可能是因为设备没有正确地初始化或配置，或者设备遇到了其他问题。需要确保device是一个有效的设备句柄，并且设备已经正确地初始化和配置。也需要检查设备是否正常工作，是否有足够的内存和电源，是否有其他程序占用了设备资源。
如果确定设备没有问题，可以尝试使用不同的超时时间，比如K4A_WAIT_INFINITE替换为一个具体的毫秒数，或者使用k4a_device_get_capture_poll()函数来轮询获取捕获对象，而不是等待。
第二天改了设备的打开方式好了。

**点云三维重建

利用点云进行三维重建，比如重建手机、电脑、椅子等物品。
