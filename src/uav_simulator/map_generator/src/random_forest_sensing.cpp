#include <pcl/point_cloud.h>//引入点云基本数据类型
#include <pcl/point_types.h>//引入点云点类型
// #include <pcl/search/kdtree.h>//旧版kd树
#include <pcl/kdtree/kdtree_flann.h>//kd树
#include <pcl_conversions/pcl_conversions.h>//pcl与ros数据转换
#include <iostream>

#include <geometry_msgs/PoseStamped.h>//引入ros消息类型:位姿
#include <geometry_msgs/Vector3.h>//引入ros消息类型:三维向量
#include <math.h>//数学库
#include <nav_msgs/Odometry.h>//引入ros消息类型:里程计
#include <ros/console.h>//ros日志调试接口
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>//引入ros消息类型:点云
#include <Eigen/Eigen>//引入Eigen库(线性代数库)
#include <random>//随机数生成器

using namespace std;

//声明全局的 kd 树用于点云最近邻搜索；同时定义两个 vector 分别存储搜索结果中的点索引和对应的距离平方值。
// pcl::search::KdTree<pcl::PointXYZ> kdtreeLocalMap;
pcl::KdTreeFLANN<pcl::PointXYZ> kdtreeLocalMap;
vector<int> pointIdxRadiusSearch;
vector<float> pointRadiusSquaredDistance;

//定义随机设备 rd 用于产生随机种子，并用该种子初始化随机数生成器 eng。注释部分显示曾用固定种子。
random_device rd;
// default_random_engine eng(4);
default_random_engine eng(rd()); 
//声明用于生成均匀分布随机数的对象，用于生成障碍物在 x、y 坐标、宽度、高度以及其它可能影响尺寸的因子。
uniform_real_distribution<double> rand_x;
uniform_real_distribution<double> rand_y;
uniform_real_distribution<double> rand_w;
uniform_real_distribution<double> rand_h;
uniform_real_distribution<double> rand_inf;

//声明 ROS 发布者和订阅者，全局变量中 _local_map_pub 用于发布局部（感知范围内）的障碍物点云，
//all_map_pub 用于发布全局点云，click_map_pub 用于发布点击时更新的点云数据；
//_odom_sub 用于接收里程计数据。
ros::Publisher _local_map_pub;
ros::Publisher _all_map_pub;
ros::Publisher click_map_pub_;
ros::Subscriber _odom_sub;

//定义一个向量存储状态信息，例如机器人位置和线速度等。
vector<double> _state;

//声明各种全局参数
int _obs_num;//障碍物数量
double _x_size, _y_size, _z_size;//地图尺寸
double _x_l, _x_h, _y_l, _y_h, _w_l, _w_h, _h_l, _h_h;//障碍物位置和尺寸范围
double _z_limit, _sensing_range, _resolution, _sense_rate, _init_x, _init_y;//感知范围、分辨率、感知频率、初始位置    
double _min_dist;//最小距离

//定义标志变量
bool _map_ok = false;//地图是否生成完成
bool _has_odom = false;//是否接收到里程计数据

int circle_num_;//圆形障碍物数量
double radius_l_, radius_h_, z_l_, z_h_;//圆形障碍物参数：半径和z轴范围
double theta_;//圆形障碍物参数：旋转角度
//定义相应的随机数分布对象
uniform_real_distribution<double> rand_radius_;
uniform_real_distribution<double> rand_radius2_;
uniform_real_distribution<double> rand_theta_;
uniform_real_distribution<double> rand_z_;

//定义 ROS 消息类型和 PCL 点云对象，用于存储全局地图、局部地图和点击生成的点云数据
sensor_msgs::PointCloud2 globalMap_pcd;
pcl::PointCloud<pcl::PointXYZ> cloudMap;
sensor_msgs::PointCloud2 localMap_pcd;
pcl::PointCloud<pcl::PointXYZ> clicked_cloud_;

//随机生成障碍物地图
//代码分为两部分：生成“极坐标”障碍物和生成“圆形”障碍物。
void RandomMapGenerate() {
  pcl::PointXYZ pt_random;
  
  //设置随机分布参数，根据参数设置各个分布对象范围，保证随机生成值在合理区间内
  rand_x = uniform_real_distribution<double>(_x_l, _x_h);
  rand_y = uniform_real_distribution<double>(_y_l, _y_h);
  rand_w = uniform_real_distribution<double>(_w_l, _w_h);
  rand_h = uniform_real_distribution<double>(_h_l, _h_h);

  rand_radius_ = uniform_real_distribution<double>(radius_l_, radius_h_);
  rand_radius2_ = uniform_real_distribution<double>(radius_l_, 1.2);
  rand_theta_ = uniform_real_distribution<double>(-theta_, theta_);
  rand_z_ = uniform_real_distribution<double>(z_l_, z_h_);

  // 生成“极坐标”障碍物
  /*
  对于每个障碍物：
  + 随机生成平面位置（x，y）和宽度（w）；
  + 按照分辨率离散化到网格中心；
  + 计算对应网格数量 widNum；
  + 使用嵌套循环在 x-y 平面上生成障碍物的横截面，再在 z 方向上从 -20 到一定高度生成点云，将每个点添加到 cloudMap 中。
  */
  for (int i = 0; i < _obs_num; i++) {
    double x, y, w, h;
    x = rand_x(eng);
    y = rand_y(eng);
    w = rand_w(eng);

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    int widNum = ceil(w / _resolution);

    for (int r = -widNum / 2.0; r < widNum / 2.0; r++)
      for (int s = -widNum / 2.0; s < widNum / 2.0; s++) {
        h = rand_h(eng);
        int heiNum = ceil(h / _resolution);
        for (int t = -20; t < heiNum; t++) {
          pt_random.x = x + (r + 0.5) * _resolution + 1e-2;
          pt_random.y = y + (s + 0.5) * _resolution + 1e-2;
          pt_random.z = (t + 0.5) * _resolution + 1e-2;
          cloudMap.points.push_back(pt_random);
        }
      }
  }

  // 生成“圆形”障碍物
  /*
  每个圆障碍物：
  + 随机生成圆心位置并离散化；
  + 生成一个随机旋转角度，并构造旋转矩阵；
  + 随机生成两个半径（radius1 和 radius2）决定椭圆形状；
  + 用循环按照角度遍历圆周，计算局部点 cpt；
  + 通过“inflate”步骤（嵌套的小循环，虽然范围只有 0）对点进行微调，再旋转、平移到全局位置后添加到 cloudMap。
  */
  for (int i = 0; i < circle_num_; ++i) {
    double x, y, z;
    x = rand_x(eng);
    y = rand_y(eng);
    z = rand_z_(eng);

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;
    z = floor(z / _resolution) * _resolution + _resolution / 2.0;

    Eigen::Vector3d translate(x, y, z);

    double theta = rand_theta_(eng);
    Eigen::Matrix3d rotate;
    rotate << cos(theta), -sin(theta), 0.0, sin(theta), cos(theta), 0.0, 0, 0,
        1;

    double radius1 = rand_radius_(eng);
    double radius2 = rand_radius2_(eng);

    // draw a circle centered at (x,y,z)
    Eigen::Vector3d cpt;
    for (double angle = 0.0; angle < 6.282; angle += _resolution / 2) {
      cpt(0) = 0.0;
      cpt(1) = radius1 * cos(angle);
      cpt(2) = radius2 * sin(angle);

      // inflate
      Eigen::Vector3d cpt_if;
      for (int ifx = -0; ifx <= 0; ++ifx)
        for (int ify = -0; ify <= 0; ++ify)
          for (int ifz = -0; ifz <= 0; ++ifz) {
            cpt_if = cpt + Eigen::Vector3d(ifx * _resolution, ify * _resolution,
                                           ifz * _resolution);
            cpt_if = rotate * cpt_if + Eigen::Vector3d(x, y, z);
            pt_random.x = cpt_if(0);
            pt_random.y = cpt_if(1);
            pt_random.z = cpt_if(2);
            cloudMap.push_back(pt_random);
          }
    }
  }

  cloudMap.width = cloudMap.points.size();
  cloudMap.height = 1;
  cloudMap.is_dense = true;

  ROS_WARN("Finished generate random map ");

  kdtreeLocalMap.setInputCloud(cloudMap.makeShared());

  _map_ok = true;
}

//生成圆柱体障碍物，步骤与上函数类似，但加入了对障碍物间距离的检查。
/*
+ 初始化随机分布（与前面类似，同时增加了 rand_inf 用于缩放因子）
+ 在生成极坐标障碍物部分中增加：
  + 用 obs_position 记录所有障碍物的（x,y）坐标，检查新障碍物若与已有障碍物距离小于 _min_dist，则跳过（减少重叠）。
  +  计算网格数量时考虑因子 inf，用于缩放宽度；并只在点处于圆形内部（距离小于半径）时才加入点云。
+ 后半部生成圆形障碍物与之前相同，最后设置 cloudMap 属性，构造 kd 树并置 _map_ok 为 true。
*/
void RandomMapGenerateCylinder() {
  pcl::PointXYZ pt_random;

  vector<Eigen::Vector2d> obs_position;

  rand_x = uniform_real_distribution<double>(_x_l, _x_h);
  rand_y = uniform_real_distribution<double>(_y_l, _y_h);
  rand_w = uniform_real_distribution<double>(_w_l, _w_h);
  rand_h = uniform_real_distribution<double>(_h_l, _h_h);
  rand_inf = uniform_real_distribution<double>(0.5, 1.5);

  rand_radius_ = uniform_real_distribution<double>(radius_l_, radius_h_);
  rand_radius2_ = uniform_real_distribution<double>(radius_l_, 1.2);
  rand_theta_ = uniform_real_distribution<double>(-theta_, theta_);
  rand_z_ = uniform_real_distribution<double>(z_l_, z_h_);

  // generate polar obs
  for (int i = 0; i < _obs_num && ros::ok(); i++) {
    double x, y, w, h, inf;
    x = rand_x(eng);
    y = rand_y(eng);
    w = rand_w(eng);
    inf = rand_inf(eng);
    
    bool flag_continue = false;
    for ( auto p : obs_position )
      if ( (Eigen::Vector2d(x,y) - p).norm() < _min_dist /*metres*/ )
      {
        i--;
        flag_continue = true;
        break;
      }
    if ( flag_continue ) continue;

    obs_position.push_back( Eigen::Vector2d(x,y) );
    

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    int widNum = ceil((w*inf) / _resolution);
    double radius = (w*inf) / 2;

    for (int r = -widNum / 2.0; r < widNum / 2.0; r++)
      for (int s = -widNum / 2.0; s < widNum / 2.0; s++) {
        h = rand_h(eng);
        int heiNum = ceil(h / _resolution);
        for (int t = -20; t < heiNum; t++) {
          double temp_x = x + (r + 0.5) * _resolution + 1e-2;
          double temp_y = y + (s + 0.5) * _resolution + 1e-2;
          double temp_z = (t + 0.5) * _resolution + 1e-2;
          if ( (Eigen::Vector2d(temp_x,temp_y) - Eigen::Vector2d(x,y)).norm() <= radius )
          {
            pt_random.x = temp_x;
            pt_random.y = temp_y;
            pt_random.z = temp_z;
            cloudMap.points.push_back(pt_random);
          }
        }
      }
  }

  // generate circle obs
  for (int i = 0; i < circle_num_; ++i) {
    double x, y, z;
    x = rand_x(eng);
    y = rand_y(eng);
    z = rand_z_(eng);

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;
    z = floor(z / _resolution) * _resolution + _resolution / 2.0;

    Eigen::Vector3d translate(x, y, z);

    double theta = rand_theta_(eng);
    Eigen::Matrix3d rotate;
    rotate << cos(theta), -sin(theta), 0.0, sin(theta), cos(theta), 0.0, 0, 0,
        1;

    double radius1 = rand_radius_(eng);
    double radius2 = rand_radius2_(eng);

    // draw a circle centered at (x,y,z)
    Eigen::Vector3d cpt;
    for (double angle = 0.0; angle < 6.282; angle += _resolution / 2) {
      cpt(0) = 0.0;
      cpt(1) = radius1 * cos(angle);
      cpt(2) = radius2 * sin(angle);

      // inflate
      Eigen::Vector3d cpt_if;
      for (int ifx = -0; ifx <= 0; ++ifx)
        for (int ify = -0; ify <= 0; ++ify)
          for (int ifz = -0; ifz <= 0; ++ifz) {
            cpt_if = cpt + Eigen::Vector3d(ifx * _resolution, ify * _resolution,
                                           ifz * _resolution);
            cpt_if = rotate * cpt_if + Eigen::Vector3d(x, y, z);
            pt_random.x = cpt_if(0);
            pt_random.y = cpt_if(1);
            pt_random.z = cpt_if(2);
            cloudMap.push_back(pt_random);
          }
    }
  }

  cloudMap.width = cloudMap.points.size();
  cloudMap.height = 1;
  cloudMap.is_dense = true;

  ROS_WARN("Finished generate random map ");

  kdtreeLocalMap.setInputCloud(cloudMap.makeShared());

  _map_ok = true;
}

/*
里程计回调函数：
+ 如果收到的里程计消息的 child_frame_id 为 “X” 或 “O” 则直接返回；
+ 否则置 _has_odom = true，并把位置和线速度保存到 _state 中（后三个分量暂设为 0）。
*/
void rcvOdometryCallbck(const nav_msgs::Odometry odom) {
  if (odom.child_frame_id == "X" || odom.child_frame_id == "O") return;
  _has_odom = true;

  _state = {odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
            0.0,
            0.0,
            0.0};
}

int i = 0;
/*
 函数 pubSensedPoints() 用于发布点云数据，
 当前只发布全局地图（cloudMap 转换成 ROS 的 PointCloud2 消息并发布到 _all_map_pub），
 代码中有注释掉的局部点云发布逻辑（基于当前状态与 kd 树的半径搜索）。
*/
void pubSensedPoints() {
  // if (i < 10) {
  pcl::toROSMsg(cloudMap, globalMap_pcd);
  globalMap_pcd.header.frame_id = "world";
  _all_map_pub.publish(globalMap_pcd);
  // }

  return;

  /* ---------- only publish points around current position ---------- */
  if (!_map_ok || !_has_odom) return;

  pcl::PointCloud<pcl::PointXYZ> localMap;

  pcl::PointXYZ searchPoint(_state[0], _state[1], _state[2]);
  pointIdxRadiusSearch.clear();
  pointRadiusSquaredDistance.clear();

  pcl::PointXYZ pt;

  if (isnan(searchPoint.x) || isnan(searchPoint.y) || isnan(searchPoint.z))
    return;

  if (kdtreeLocalMap.radiusSearch(searchPoint, _sensing_range,
                                  pointIdxRadiusSearch,
                                  pointRadiusSquaredDistance) > 0) {
    for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
      pt = cloudMap.points[pointIdxRadiusSearch[i]];
      localMap.points.push_back(pt);
    }
  } else {
    ROS_ERROR("[Map server] No obstacles .");
    return;
  }

  localMap.width = localMap.points.size();
  localMap.height = 1;
  localMap.is_dense = true;

  pcl::toROSMsg(localMap, localMap_pcd);
  localMap_pcd.header.frame_id = "world";
  _local_map_pub.publish(localMap_pcd);
}

/*
此回调函数响应某个点击消息：
+ 获取点击位置（x, y）；离散化处理；
+ 根据随机宽度和高度生成新的点块，同时添加到 clicked_cloud_ 和全局 cloudMap；
+ 更新 clicked_cloud_ 的点云属性后转换为 ROS 消息，发布到 click_map_pub_。
*/
void clickCallback(const geometry_msgs::PoseStamped& msg) {
  double x = msg.pose.position.x;
  double y = msg.pose.position.y;
  double w = rand_w(eng);
  double h;
  pcl::PointXYZ pt_random;

  x = floor(x / _resolution) * _resolution + _resolution / 2.0;
  y = floor(y / _resolution) * _resolution + _resolution / 2.0;

  int widNum = ceil(w / _resolution);

  for (int r = -widNum / 2.0; r < widNum / 2.0; r++)
    for (int s = -widNum / 2.0; s < widNum / 2.0; s++) {
      h = rand_h(eng);
      int heiNum = ceil(h / _resolution);
      for (int t = -1; t < heiNum; t++) {
        pt_random.x = x + (r + 0.5) * _resolution + 1e-2;
        pt_random.y = y + (s + 0.5) * _resolution + 1e-2;
        pt_random.z = (t + 0.5) * _resolution + 1e-2;
        clicked_cloud_.points.push_back(pt_random);
        cloudMap.points.push_back(pt_random);
      }
    }
  clicked_cloud_.width = clicked_cloud_.points.size();
  clicked_cloud_.height = 1;
  clicked_cloud_.is_dense = true;

  pcl::toROSMsg(clicked_cloud_, localMap_pcd);
  localMap_pcd.header.frame_id = "world";
  click_map_pub_.publish(localMap_pcd);

  cloudMap.width = cloudMap.points.size();

  return;
}

void generate_simple()
{
  pcl::PointXYZ pt_random;

  // 固定障碍物参数
  double x = 0.0;  // 障碍物在原点
  double y = 0.0;
  double radius = 1.0;  // 1米半径
  double height = 2.0;  // 2米高

  // 将坐标对齐到分辨率网格
  x = floor(x / _resolution) * _resolution + _resolution / 2.0;
  y = floor(y / _resolution) * _resolution + _resolution / 2.0;

  int widNum = ceil((2 * radius) / _resolution);  // 计算网格数量

  // 生成圆柱体障碍物
  for (int r = -widNum / 2.0; r < widNum / 2.0; r++) {
    for (int s = -widNum / 2.0; s < widNum / 2.0; s++) {
      int heiNum = ceil(height / _resolution);
      for (int t = 0; t < heiNum; t++) {  // 从地面开始
        double temp_x = x + (r + 0.5) * _resolution + 1e-2;
        double temp_y = y + (s + 0.5) * _resolution + 1e-2;
        double temp_z = (t + 0.5) * _resolution + 1e-2;
        
        // 只在圆柱体范围内添加点
        if ((Eigen::Vector2d(temp_x,temp_y) - Eigen::Vector2d(x,y)).norm() <= radius) {
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
      }
    }
  }

  cloudMap.width = cloudMap.points.size();
  cloudMap.height = 1;
  cloudMap.is_dense = true;

  ROS_WARN("Finished generate single cylinder obstacle");

  kdtreeLocalMap.setInputCloud(cloudMap.makeShared());
  _map_ok = true;
}

int main(int argc, char** argv) {
  /*
  初始化 ROS 节点，命名为 "random_map_sensing"，创建私有命名空间的 NodeHandle。
  */
  ros::init(argc, argv, "random_map_sensing");
  ros::NodeHandle n("~");
  /*
  初始化三个发布者（局部地图、全局地图和点击地图）以及一个里程计消息的订阅者。点击回调的订阅被注释掉。
  */
  _local_map_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/local_cloud", 1);
  _all_map_pub = n.advertise<sensor_msgs::PointCloud2>("/map_generator/global_cloud", 1);

  _odom_sub = n.subscribe("odometry", 50, rcvOdometryCallbck);

  click_map_pub_ =
      n.advertise<sensor_msgs::PointCloud2>("/pcl_render_node/local_map", 1);
  // ros::Subscriber click_sub = n.subscribe("/goal", 10, clickCallback);
  
  //参数读取：从参数服务器读取各种参数，
  //包括初始状态、地图尺寸、障碍物数量、障碍物尺寸范围、障碍物形状参数以及传感器参数。
  //提供默认值以防参数不存在。
  n.param("init_state_x", _init_x, 0.0);
  n.param("init_state_y", _init_y, 0.0);

  n.param("map/x_size", _x_size, 20.0);
  n.param("map/y_size", _y_size, 20.0);
  n.param("map/z_size", _z_size, 3.0);
  n.param("map/obs_num", _obs_num, 1);
  n.param("map/resolution", _resolution, 0.1);
  n.param("map/circle_num", circle_num_, 30);

  n.param("ObstacleShape/lower_rad", _w_l, 0.3);
  n.param("ObstacleShape/upper_rad", _w_h, 0.8);
  n.param("ObstacleShape/lower_hei", _h_l, 3.0);
  n.param("ObstacleShape/upper_hei", _h_h, 7.0);

  n.param("ObstacleShape/radius_l", radius_l_, 7.0);
  n.param("ObstacleShape/radius_h", radius_h_, 7.0);
  n.param("ObstacleShape/z_l", z_l_, 7.0);
  n.param("ObstacleShape/z_h", z_h_, 7.0);
  n.param("ObstacleShape/theta", theta_, 7.0);

  n.param("sensing/radius", _sensing_range, 10.0);
  n.param("sensing/rate", _sense_rate, 10.0);

  n.param("min_distance", _min_dist, 1.0);

  //计算一些辅助变量
  _x_l = -_x_size / 2.0;
  _x_h = +_x_size / 2.0;

  _y_l = -_y_size / 2.0;
  _y_h = +_y_size / 2.0;

  //根据地图尺寸计算 x, y 坐标的下限和上限；同时对障碍物数量做上限限制（不超过 x_size*10）。
  _obs_num = min(_obs_num, (int)_x_size * 10);
  _z_limit = _z_size;

  
  ros::Duration(0.5).sleep();

  //等待一段时间后种子设置
  //暂停 0.5 秒以保证参数加载完成；获取随机种子并输出，重新设置随机数生成器种子。
  unsigned int seed = rd();
  // unsigned int seed = 2433201515;
  cout << "seed=" << seed << endl;
  eng.seed(seed);

  // RandomMapGenerate();
  //生成地图
  // RandomMapGenerateCylinder();
  generate_simple();

  ros::Rate loop_rate(_sense_rate);

  //进入循环：按照传感器频率发布点云数据
  //（通过 pubSensedPoints()），处理回调（ros::spinOnce()），并等待一段时间以保证固定发布频率
  while (ros::ok()) {
    pubSensedPoints();
    ros::spinOnce();
    loop_rate.sleep();
  }
}