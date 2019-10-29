#include "v9_localization/v9_localization.h"

#ifdef DEBUG
int debug_viz_mode = 0;
#endif

const int Localization::MIN_LINE_INLIERS = 10;
const int Localization::MIN_CIRCLE_INLIERS = 30;
const int Localization::MIN_LINE_LENGTH = 60;
const int Localization::MAX_LINE_MODEL = 10;

Localization::Localization():
    nh_(ros::this_node::getName()),it_(nh_),
    sw_sub_(nh_, "/v9_ball_detector_node/segment_white",1),
    isg_sub_(nh_,"/v9_ball_detector_node/inv_segment_green",1),
    fb_sub_(nh_,"/v9_ball_detector_node/field_boundary",1),
    js_sub_(nh_,"/robotis/present_joint_states",1),
    #ifdef GAZEBO
    imu_sub_(nh_,"/robotis_op3/imu",1),
    #else
    imu_sub_(nh_,"/arduino_controller/imu",1),
    #endif
    input_sync_(SyncPolicy(10), sw_sub_,isg_sub_,fb_sub_,js_sub_,imu_sub_),
    robot_state_pub_(nh_.advertise<geometry_msgs::PoseStamped > ("robot_state", 10)),
    particles_state_pub_(nh_.advertise<vision_utils::Particles> ("particles_state", 10)),
    features_pub_(nh_.advertise<vision_utils::Features > ("field_features", 10)),
    //present_joint_sub_ = nh_.subscribe("/robotis/present_joint_states",100,&Localization::presentJointStateCb,this);
    #ifdef GAZEBO
    //imu_data_sub_(nh_.subscribe("/robotis_op3/imu",1,&Localization::imuDataCb,this)),
    camera_info_sub_(nh_.subscribe("/robotis_op3/camera/camera_info",1,&Localization::cameraInfoCallback,this)),
    #else
    //imu_data_sub_ = nh_.subscribe("/arduino_controller/imu",1,&Localization::imuDataCb,this);
    camera_info_sub_(nh_.subscribe("/usb_cam/camera_info", 1, &Localization::cameraInfoCallback, this)),
    #endif
    reset_particles_sub_(nh_.subscribe("/localization_monitor_node/reset_particles", 1, &Localization::resetParticlesCb, this)),
    save_params_sub_(nh_.subscribe("/vision_monitor_node/save_param", 1, &Localization::saveParamsCb, this)),
    ball_pos_sub_(nh_.subscribe("/v9_ball_detector_node/ball_pos",1,&Localization::ballPosCb,this)),
    projected_ball_pub_(nh_.advertise<geometry_msgs::Point>("projected_ball",10)),
    projected_ball_stamped_pub_(nh_.advertise<geometry_msgs::PointStamped >("stamped_projected_ball", 10)),
    line_tip_pub_(nh_.advertise<vision_utils::LineTip> ("line_tip", 10)),
    #ifdef GAZEBO
    odometry_sub_(nh_.subscribe("/robotis/pelvis_pose", 1, &Localization::odometryCb, this)),
    #else
    odometry_sub_(nh_.subscribe("/alfarobi/odometry", 100, &Localization::odometryCb, this)),
    robot_height_sub_(nh_.subscribe("/alfarobi/robot_height", 100, &Localization::robotHeightCb, this)),
    #endif
    sw_encoding_(Alfarobi::GRAY8Bit),
    isg_encoding_(Alfarobi::GRAY8Bit),
    reset_particles_req_(false),
    lost_features_(true),
    robot_height_(48.0f),
    gy_heading_(.0f),
    last_gy_heading_(.0f),
    imu_data_{.0, .0, .0},
    front_fall_limit_(.0),behind_fall_limit_(.0),
    right_fall_limit_(.0),left_fall_limit_(.0){

    nh_.param<double>("H_FOV",H_FOV, 61.25);
    nh_.param<double>("V_FOV",V_FOV, 47.88);
    nh_.param<double>("circle_cost", circle_cost, 6.0);
    nh_.param<double>("inlier_error", inlier_error, 1.0);
    nh_.param<double>("fx",fx_, 540.552005478);
    nh_.param<double>("fy",fy_, 540.571602012);
    nh_.param<double>("roll_offset",roll_offset_, .0);
    nh_.param<double>("pitch_offset",pitch_offset_, .0);
    nh_.param<double>("yaw_offset",yaw_offset_, .0);
    nh_.param<double>("tilt_limit",tilt_limit_, 30.0);
    nh_.param<double>("z_offset", z_offset_, .0);
    nh_.param<double>("pan_rot_comp", pan_rot_comp_, .0);
    nh_.param<double>("shift_const", shift_const_, -240.0);
    nh_.param<bool>("attack_dir", attack_dir_, false);

    roll_offset_ *= Math::DEG2RAD;
    pitch_offset_ *= Math::DEG2RAD;
    tilt_limit_ *= Math::DEG2RAD;
    H_FOV *= Math::DEG2RAD;
    V_FOV *= Math::DEG2RAD;
    pan_rot_comp_ *= Math::DEG2RAD;

    TAN_HFOV_PER2 = tan(H_FOV * 0.5);
    TAN_VFOV_PER2 = tan(V_FOV * 0.5);

    input_sync_.registerCallback(boost::bind(&Localization::utilsCallback, this, _1, _2, _3, _4, _5));

    param_cb_ = boost::bind(&Localization::paramCallback, this, _1, _2);
    server_.setCallback(param_cb_);
    params_req_ = false;

    loadParams();
    initializeParticles();
    initializeFieldFeaturesData();    
    genRadialPattern();
    initializeFK();
#ifdef DEBUG
    cv::namedWindow("DEBUG_VIZ");
    cv::createTrackbar("VIZ_MODE", "DEBUG_VIZ", &debug_viz_mode,5);
#endif
}

Localization::~Localization(){

}

//void Localization::presentJointStateCb(const sensor_msgs::JointStateConstPtr &_msg){

//}

void Localization::utilsCallback(const sensor_msgs::ImageConstPtr &_sw_msg,
                                 const sensor_msgs::ImageConstPtr &_isg_msg,
                                 const vision_utils::FieldBoundary::ConstPtr &_fb_msg,
                                 const sensor_msgs::JointStateConstPtr &_js_msg,
                                 const sensor_msgs::ImuConstPtr &_imu_msg){
    try{
        //[HW] : Fix encoding type
        sw_encoding_ = (_sw_msg->encoding.compare(sensor_msgs::image_encodings::MONO8))?Alfarobi::BGR8Bit:Alfarobi::GRAY8Bit;
        isg_encoding_ = (_isg_msg->encoding.compare(sensor_msgs::image_encodings::MONO8))?Alfarobi::BGR8Bit:Alfarobi::GRAY8Bit;
    }catch(cv_bridge::Exception &e){
        ROS_ERROR("[v9_localization] cv_bridge exception: %s",e.what());
    }

    cv_sw_ptr_sub_ = cv_bridge::toCvCopy(_sw_msg);
    cv_isg_ptr_sub_ = cv_bridge::toCvCopy(_isg_msg);
    field_boundary_.bound1 = _fb_msg->bound1;
    field_boundary_.bound2 = _fb_msg->bound2;

    int head_count = 0;
    for(size_t i=0;i<_js_msg->name.size() && head_count < 2;i++){
        if(_js_msg->name[i] == "head_pan"){
            pan_servo_angle_ = (_js_msg->position[i] * -1) + pan_servo_offset_;
            ++head_count;
        }
        if(_js_msg->name[i] == "head_tilt"){
#ifdef GAZEBO
            tilt_servo_angle_ = (_js_msg->position[i] * -1) - tilt_servo_offset_;
#else
            tilt_servo_angle_ = _js_msg->position[i] - tilt_servo_offset_; //from offset tuner
#endif
            ++head_count;
        }
    }

    Eigen::Quaterniond orientation;

    orientation.x() = _imu_msg->orientation.x;
    orientation.y() = _imu_msg->orientation.y;
    orientation.z() = _imu_msg->orientation.z;
    orientation.w() = _imu_msg->orientation.w;
    imu_data_ = robotis_framework::convertQuaternionToRPY(orientation);
#ifdef GAZEBO
    roll_compensation_ = -imu_data_.coeff(0) + roll_offset_;
    offset_head_ = imu_data_.coeff(1) + pitch_offset_;
//    offset_head_ = imu_data_.coeff(1);
    gy_heading_ = -imu_data_.coeff(2) * Math::RAD2DEG;
    if(gy_heading_ < 0)gy_heading_ = 360.0f + gy_heading_;
    odometer_out_.theta = (gy_heading_ - last_gy_heading_)*Math::DEG2RAD;
    last_gy_heading_ = gy_heading_;
#else
    roll_compensation_ = imu_data_.coeff(0) + roll_offset_;
    offset_head_ = -imu_data_.coeff(1) + pitch_offset_;
    gy_heading_ = -imu_data_.coeff(2) * Math::RAD2DEG + yaw_offset_;
    //gy_heading_ = 360 - gy_heading_;// UNTUK ROBI YAW-NYA KEBALIK
    if(gy_heading_ < .0f)gy_heading_ = 360.0f + gy_heading_;
    odometer_out_.theta = (gy_heading_ - last_gy_heading_)*Math::DEG2RAD;
    last_gy_heading_ = gy_heading_;
#endif

    // Get header from one of the message
    this->stamp_ = _sw_msg->header.stamp;
    this->frame_id_ = _sw_msg->header.frame_id;
}

void Localization::paramCallback(v9_localization::LocalizationParamsConfig &_config, uint32_t level){
    (void)level;
    new_params_ = _config;
    params_req_ = true;
}

void Localization::actionForMonitor(){
    if(params_req_){
        if(params_.num_particles != new_params_.num_particles){
            particles_state_.resize(new_params_.num_particles);
            if(params_.num_particles < new_params_.num_particles){
                float uniform_weight = 1.0f/new_params_.num_particles;
                for(size_t i=params_.num_particles;i<new_params_.num_particles;i++){
                    particles_state_[i].x = genUniIntDist(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_LENGTH);
                    particles_state_[i].y = genUniIntDist(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_WIDTH);
                    particles_state_[i].z = genUniIntDist(0, 360);
                    particles_state_[i].w = uniform_weight;
                }
            }
        }
        params_ = new_params_;
        params_req_ = false;
    }

    if(reset_particles_req_){        
        initializeParticles();
        reset_particles_req_ = false;
    }
}

void Localization::publishTipPoints(vision_utils::LineTip _tip_points){
    line_tip_pub_.publish(_tip_points);
}

bool Localization::setInputImage(){

    if(cv_sw_ptr_sub_ != nullptr && cv_isg_ptr_sub_ != nullptr){
        
        segmented_white_ = cv_sw_ptr_sub_->image;
        invert_green_ = cv_isg_ptr_sub_->image;

        //[HW] : Fix encoding type
//        if(sw_encoding_ == Alfarobi::BGR8Bit)
//            cv::cvtColor(segmented_white_,segmented_white_,CV_BGR2GRAY);
//        if(isg_encoding_ == Alfarobi::BGR8Bit)
//            cv::cvtColor(invert_green_, invert_green_,CV_BGR2GRAY);

//        cv::erode(invert_green_,invert_green_,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));
//        cv::dilate(invert_green_,invert_green_,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));
//        cv::dilate(invert_green_,invert_green_,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));
//        cv::erode(invert_green_,invert_green_,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3)));

        return true;
    }else{
        return false;
    }
}

void Localization::initializeFieldFeaturesData(){
    landmark_pos_.resize(4);
    line_segment_pos_.resize(11);
    //L - landmark
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH)*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_WIDTH)*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH - GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH - GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH)*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH + FIELD_WIDTH)*0.01f);
    //T - landmark
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH)*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + FIELD_WIDTH)*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    //X - landmark
    landmark_pos_[2].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + ((FIELD_WIDTH - CENTER_CIRCLE_DIAMETER)>>1))*0.01f);
    landmark_pos_[2].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + ((FIELD_WIDTH + CENTER_CIRCLE_DIAMETER)>>1))*0.01f);
    //Center circle
    landmark_pos_[3].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + (FIELD_WIDTH>>1))*0.01f);

    //Vertical Line Segment
    line_segment_pos_[0] = {landmark_pos_[0][0].x,landmark_pos_[0][0].y,landmark_pos_[0][1].x,landmark_pos_[0][1].y};
    line_segment_pos_[1] = {landmark_pos_[0][2].x,landmark_pos_[0][2].y,landmark_pos_[0][3].x,landmark_pos_[0][3].y};
    line_segment_pos_[2] = {landmark_pos_[1][2].x,landmark_pos_[1][2].y,landmark_pos_[1][3].x,landmark_pos_[1][3].y};
    line_segment_pos_[3] = {landmark_pos_[0][4].x,landmark_pos_[0][4].y,landmark_pos_[0][5].x,landmark_pos_[0][5].y};
    line_segment_pos_[4] = {landmark_pos_[0][6].x,landmark_pos_[0][6].y,landmark_pos_[0][7].x,landmark_pos_[0][7].y};
    //Horizontal Line Segment
    line_segment_pos_[5] = {landmark_pos_[0][0].x,landmark_pos_[0][0].y,landmark_pos_[0][6].x,landmark_pos_[0][6].y};
    line_segment_pos_[6] = {landmark_pos_[1][0].x,landmark_pos_[1][0].y,landmark_pos_[0][2].x,landmark_pos_[0][2].y};
    line_segment_pos_[7] = {landmark_pos_[0][4].x,landmark_pos_[0][4].y,landmark_pos_[1][4].x,landmark_pos_[0][4].y};
    line_segment_pos_[8] = {landmark_pos_[1][1].x,landmark_pos_[1][1].y,landmark_pos_[0][3].x,landmark_pos_[0][3].y};
    line_segment_pos_[9] = {landmark_pos_[0][5].x,landmark_pos_[0][5].y,landmark_pos_[1][5].x,landmark_pos_[1][5].y};
    line_segment_pos_[10] = {landmark_pos_[0][1].x,landmark_pos_[0][1].y,landmark_pos_[0][7].x,landmark_pos_[0][7].y};
}

inline float Localization::panAngleDeviation(float _pixel_x_pos){
    return atan((2.0f * _pixel_x_pos/FRAME_WIDTH - 1.0f) * TAN_HFOV_PER2);
}

inline float Localization::tiltAngleDeviation(float _pixel_y_pos){
    return atan((2.0f * _pixel_y_pos/FRAME_HEIGHT - 1.0f) * TAN_VFOV_PER2);
}

inline float Localization::verticalDistance(float _tilt_dev){
    float total_tilt = Math::PI_TWO - (CAMERA_ORIENTATION.coeff(1) + _tilt_dev);
    return (robot_height_ + CAMERA_DIRECTION.coeff(2) + z_offset_) * tan(total_tilt);
}

inline float Localization::horizontalDistance(float _distance_y, float _offset_pan){
    return _distance_y * tan(_offset_pan);
}

void Localization::genRadialPattern(){
    for(int i = 1; i <= 10; i++){
        float angle_step = 360.0f/(8.0f*(float)i) * Math::DEG2RAD;
//        std::cout << "========================================" << std::endl;
        int total_nb = 8*i;
        for(int j=0;j<total_nb;j++){
            std::pair<int,int > nb_pattern;
            int radius=i;
            if(j <= total_nb/8)
                radius /= cos(angle_step*(float)j);
            else if (j <= (3*total_nb)/8)
                radius /= sin(angle_step*(float)j);
            else if (j <= (5*total_nb)/8)
                radius /= -cos(angle_step*(float)j);
            else if (j < (7*total_nb)/8)
                radius /= -sin(angle_step*(float)j);
            else
                radius /= cos(angle_step*(float)j);

            float est_x = (float)radius * cos(angle_step*(float)j);
            float est_y = (float)(-radius) * sin(angle_step*(float)j);
            // numerical error
            nb_pattern.first = est_x < -1e-4f ? std::floor(est_x):
                                                (est_x > 1e-4f ? std::ceil(est_x):std::abs(est_x));
            nb_pattern.second = est_y < -1e-4f ? std::floor(est_y):
                                                 (est_y > 1e-4f ? std::ceil(est_y):std::abs(est_y));
//            std::cout << est_x << " ; " << est_y << std::endl;
//            std::cout << nb_pattern.first << " ; " << nb_pattern.second << std::endl;
            radial_pattern_.push_back(nb_pattern);
        }
    }
}

//void Localization::resetParticles(Particles &_particles){
//    _particles.resize(params_.num_particles);
//    float uniform_weight = 1.0/params_.num_particles;
//    for(Particles::iterator it=_particles.begin();
//        it!=_particles.end();it++){
//        it->x = genUniIntDist(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_LENGTH);
//        it->y = genUniIntDist(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_WIDTH);
//        it->z = genUniIntDist(0, 360);
//        it->w = uniform_weight;
//    }
//}

void Localization::sampleMotionModelOdometry(
        Particles &_particles_state,
        const geometry_msgs::Pose2D &_odometer_out){
    bool idle = odometer_buffer_.size() == 0;
    
    float diff_x = .0f;
    float diff_y = .0f;
    
    if(!idle){
        diff_x = odometer_buffer_.front().x;
        diff_y = -odometer_buffer_.front().y;
    }

    float drot1 = atan2(diff_x, -diff_y);// + gy_heading_*Math::DEG2RAD;
    float dtrans = sqrt(diff_x*diff_x + diff_y*diff_y);
    float drot2 = _odometer_out.theta; //- drot1;

    //temporarily not yet use motion noise
//    float drot1_sqr = drot1*drot1;
//    float dtrans_sqr = dtrans*dtrans;
//    float drot2_sqr = drot2*drot2;

    double noise_std_dev = features_present_ > 0 ? 1.75 : .0;//std::min(3,features_present_);

    for(Particles::iterator it=_particles_state.begin();
        it!=_particles_state.end();it++){
//        if(it->z < 0)it->z = 360 + it->z;
        float drot1_hat = drot1 ;//- sampleNormal(params_.alpha1*drot1_sqr + params_.alpha2*dtrans_sqr);
        float dtrans_hat = dtrans ;//- sampleNormal(params_.alpha3*dtrans_sqr + params_.alpha4*drot1_sqr + params_.alpha4*drot2_sqr);
        float drot2_hat = drot2 ;//- sampleNormal(params_.alpha1*drot2_sqr + params_.alpha2*dtrans_sqr);
        drot1_hat *= Math::RAD2DEG;
        drot2_hat *= Math::RAD2DEG;

//        if(dtrans_hat > 0.0)std::cout << "ROT1_HAT : " << drot1_hat << " ; TRANS_HAT : " << dtrans_hat << " ; ROT2_HAT : " << drot2_hat << std::endl;
        float tetha = (it->z /*+ drot1_hat*/)*Math::DEG2RAD;
        //Posterior Pose
//        it->x = it->x + dtrans_hat*cos(tetha) + features_present_ * (idle ? genNormalDist(0,0.5) : 0);
//        it->y = it->y + dtrans_hat*sin(tetha) + features_present_ * (idle ? genNormalDist(0,0.5) : 0);
//        it->z = it->z + /*drot1_hat +*/ drot2_hat + features_present_ * (idle ? genNormalDist(0,1) : 0);
        it->x = it->x + dtrans_hat*cos(tetha) + genNormalDist(.0, noise_std_dev);
        it->y = it->y + dtrans_hat*sin(tetha) + genNormalDist(.0, noise_std_dev);
        it->z = it->z + /*drot1_hat +*/ drot2_hat + genNormalDist(.0, noise_std_dev);
        if(it->z < .0)it->z = 360.0 + it->z;
    }

    odometer_out_.theta = .0;
}

void Localization::measurementModel(Particles &_particles_state,
                                    vision_utils::Features _features_arg,
                                    float &_weight_avg){
    if(_features_arg.feature.size() == 0){
//        float uniform_weight = 1.0/params_.num_particles;
//        for(Particles::iterator it = _particles_state.begin();
//            it != _particles_state.end(); it++){
//            it->w = uniform_weight;
//        }
        // _weight_avg = uniform_weight;
        _weight_avg = last_weight_avg_;
        return;
    }

    vision_utils::Features _features = _features_arg;
    for(std::vector<vision_utils::Feature>::iterator it = _features.feature.begin();
        it != _features.feature.end(); it++){
        it->param1 *= .01f;
        it->param2 *= .01f;
        it->param3 *= .01f;
        it->param4 *= .01f;

//        std::cout << "Param 4 : " << it->param4 << std::endl;
//        std::cout << "ORIENTATION : "  << it->orientation  << std::endl;
        it->orientation *= Math::DEG2RAD;
    }

    float total_weight = .0f;
//     float minimum_weight = std::numeric_limits<float>::max();
//    float max_weight = std::numeric_limits<float>::min();
//    int num_features = _features.feature.size();
//    float max_weight = 0;
    cv::Vec3f top3_weight = {.0f, .0f, .0f};
    bool acquisition[3] = {false, false, false};
    //Range, Beam, Correspondence
    Vecs4 weight_param(_features.feature.size());
    for(Particles::iterator it = _particles_state.begin();
        it != _particles_state.end(); it++){
//        std::cout << it->z << std::endl;        

//        bool segline_used[11] = {false,false,false,false,false,
//                                 false,false,false,false,false,false};
        float pos_x = it->x * .01f;
        float pos_y = it->y * .01f;
        float tetha =  it->z * Math::DEG2RAD;
        float c_t = cos(tetha);
        float s_t = sin(tetha);
        for(std::pair<std::vector<vision_utils::Feature>::iterator,
            Vecs4::iterator > it_pair(_features.feature.begin(), weight_param.begin());
            it_pair.first != _features.feature.end();
            it_pair.first++, it_pair.second++){

            int feature_type = it_pair.first->feature_type;
            (*it_pair.second)[0] = FIELD_LENGTH;
            (*it_pair.second)[1] = Math::PI_TWO;
            (*it_pair.second)[2] = .0f;
            (*it_pair.second)[3] = -1.0f; // Unknown Feature

            if(feature_type < 4){//L, T, X, circle landmark                
                float optimal_diff = std::numeric_limits<float>::max();
                for(size_t j = 0; j< landmark_pos_[feature_type].size(); j++){
                    float delta_x = (landmark_pos_[feature_type][j].x - pos_x);
                    float delta_y = (landmark_pos_[feature_type][j].y - pos_y);
                    float feature_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                    float diff = std::fabs(feature_dist - it_pair.first->param4);
                    if(diff < optimal_diff){                        
                        (*it_pair.second)[0] = diff;
                        float beam_dev = atan2(delta_y,delta_x);
                        int map_tetha = tetha > Math::PI ? (tetha-Math::TWO_PI) : tetha;
                        beam_dev = beam_dev - map_tetha;
//                        if(beam_dev < 0)beam_dev = Math::TWO_PI + beam_dev;
//                        beam_dev = tetha-beam_dev;
//                        beam_dev = tetha-beam_dev;
//                        int dir=1;
//                        if(fabs(beam_dev) > 180)dir =- 1;
//                        beam_dev = std::min(beam_dev, 360 - beam_dev) * dir;
//                        (*it_pair.second)[1] = fabs((beam_dev < Math::PI ? beam_dev : Math::TWO_PI - beam_dev) - it_pair.first->orientation);
                        beam_dev = std::fabs(beam_dev - it_pair.first->orientation);
                        (*it_pair.second)[1] = beam_dev;
                        (*it_pair.second)[2] = .0f;
                        (*it_pair.second)[3] = -1.0f;
//                        std::cout << (*it_pair.second) << std::endl;
                        optimal_diff = diff;
                    }

                }
            }else{
                float optimal_diff = std::numeric_limits<float>::max();

                cv::Point2f a(pos_x + it_pair.first->param2*c_t - it_pair.first->param1*s_t,
                            pos_y + it_pair.first->param2*s_t + it_pair.first->param1*c_t);
                cv::Point2f b(pos_x + it_pair.first->param4*c_t - it_pair.first->param3*s_t,
                            pos_y + it_pair.first->param4*s_t + it_pair.first->param3*c_t);
//                float segline_len = sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
                float orientation2V = std::min(std::fabs(Math::PI_TWO - tetha),
                                               std::fabs(Math::THREE_PI_TWO - tetha));
                float orientation2H = std::min(std::fabs(tetha),
                                      std::min(std::fabs(Math::TWO_PI - tetha),
                                               std::fabs(Math::PI - tetha)));

                float diffV = std::fabs(orientation2V - it_pair.first->orientation);
                float diffH = std::fabs(orientation2H - it_pair.first->orientation);

                for(size_t j = 0;j < line_segment_pos_.size(); j++){
//                    if(segline_used[j])continue;
                    if(j < 5){//Vertical Index
//                        if(a.y > line_segment_pos_[j][1] && b.y < line_segment_pos_[j][3]){
//                        float refline_len = (line_segment_pos_[j][3] - line_segment_pos_[j][1]);
//                        if(segline_len <= refline_len &&
//                            (std::min(a.y,b.y) - line_segment_pos_[j][1]) > -MAX_DIFF_OFFSET && // maximum 25 cm offset
//                            (std::max(a.y,b.y) - line_segment_pos_[j][3]) < MAX_DIFF_OFFSET){
                        float diff_offset1 = std::max(.0f, line_segment_pos_[j][1] - std::min(a.y,b.y));
                        float diff_offset2 = std::max(.0f, std::max(a.y,b.y) - line_segment_pos_[j][3]);
//                        float diff = std::fabs(a.x - line_segment_pos_[j][2]) + std::fabs(b.x - line_segment_pos_[j][0]);
                        float diff = std::fabs((std::max(a.x,b.x) - line_segment_pos_[j][2]) + (std::min(a.x,b.x) - line_segment_pos_[j][0]));
//                        float diff = std::fabs(std::max(a.x,b.x) - line_segment_pos_[j][2]) - std::fabs(std::min(a.x,b.x) - line_segment_pos_[j][0]);
                        float temp_diff = diff + diff_offset1 + diff_offset2 + diffV;
                        if(temp_diff < optimal_diff){
                            optimal_diff = temp_diff;
                            (*it_pair.second)[0] = diff + diff_offset1 + diff_offset2;
//                            float beam_dev = std::fabs(orientation_p2ref - it_pair.first->orientation);
                            (*it_pair.second)[1] = diffV;//fabs(beam_dev < 90.0 ? beam_dev : 180-beam_dev);
                            (*it_pair.second)[2] = diff_offset1 + diff_offset2;
                            (*it_pair.second)[3] = j;
//                            }
                        }
                    }else{//Horizontal Remains
//                        if(a.x > line_segment_pos_[j][0] && b.x < line_segment_pos_[j][2]){
//                        float refline_len = (line_segment_pos_[j][2] - line_segment_pos_[j][0]);
//                        if(segline_len <= refline_len ){
                        float diff_offset1 = std::max(.0f, line_segment_pos_[j][0] - std::min(a.x,b.x));
                        float diff_offset2 = std::max(.0f, std::max(a.x,b.x) - line_segment_pos_[j][2]);
//                        float diff = std::fabs(a.y - line_segment_pos_[j][3]) + std::fabs(b.y - line_segment_pos_[j][1]);
                        float diff = std::fabs((std::max(a.y,b.y) - line_segment_pos_[j][3]) + (std::min(a.y,b.y) - line_segment_pos_[j][1]));
//                        float diff = std::fabs(std::max(a.y,b.y) - line_segment_pos_[j][3]) - std::fabs(std::min(a.y,b.y) - line_segment_pos_[j][1]);
                        float temp_diff = diff + diff_offset1 + diff_offset2 + diffH;
                        if(temp_diff < optimal_diff){
                            optimal_diff = temp_diff;
                            (*it_pair.second)[0] = diff + diff_offset1 + diff_offset2;
//                            float beam_dev = std::fabs(orientation - it_pair.first->orientation);
                            (*it_pair.second)[1] = diffH;//fabs(beam_dev < 180.0 ? 90 - beam_dev : 270 - beam_dev);
                            (*it_pair.second)[2] = diff_offset1 + diff_offset2;
                            (*it_pair.second)[3] = j;
                        }
//                        }
                    }
                }
//                if((*it_pair.second)[2]>0)segline_used[(int)(*it_pair.second)[2]] = true;
            }

        }

        int features_used = 0;

        for(Vecs4::const_iterator it2 = weight_param.begin();
            it2 != weight_param.end(); it2++){
//            std::cout << (*it2)[0] << " ; " << (*it2)[1] << " ; " << (*it2)[3] << std::endl;
            float weight = (pdfRange((*it2)[0])*pdfBeam((*it2)[1]));//*probDensityFunc((*it2)[2],1.0));

            if(weight > top3_weight[0] && !acquisition[0]){
                acquisition[0] = (*it2)[3] == -1.0f;
                top3_weight[1] = top3_weight[0];
                top3_weight[0] = weight;
                ++features_used;
            }else if(weight > top3_weight[1] && !acquisition[1]){
                acquisition[1] = (*it2)[3] == -1.0f;
                top3_weight[2] = top3_weight[1];
                top3_weight[1] = weight;
                ++features_used;
            }else if(weight > top3_weight[2] && !acquisition[2]){
                acquisition[2] = (*it2)[3] == -1.0f;
                top3_weight[2] = weight;
                ++features_used;
            }

//            updated_weight += weight;
//            it->w = updated_weight;
        }
        features_used = std::max(1, std::min(features_used, 3));

        float gy_dev = fabs(tetha - gy_heading_ * Math::DEG2RAD);
        gy_dev = gy_dev > Math::PI ? Math::TWO_PI - gy_dev : gy_dev;
        float updated_weight = (top3_weight[0]*(int)(features_used > 0) +
                          top3_weight[1]*(int)(features_used > 1) +
                          top3_weight[2]*(int)(features_used > 2))/features_used;
//        std::cout << "VIS WEIGHT : " << updated_weight << std::endl;
        // float gy_weight = probDensityFunc(gy_dev,params_.gy_var);
        float gy_weight = expWeight(gy_dev, params_.gy_var);
//        std::cout << "GY Weight : " << gy_weight << std::endl;
        updated_weight *= gy_weight;

        it->w = updated_weight;

//        if(updated_weight > max_weight)max_weight = updated_weight;

        total_weight += updated_weight;

//         if(updated_weight < minimum_weight)
//             minimum_weight = updated_weight;

//        if(updated_weight > max_weight){
//            max_weight = updated_weight;
//        }

        top3_weight[0] = top3_weight[1] = top3_weight[2] = .0f;
        acquisition[0] = acquisition[1] = acquisition[2] = false;
    }
#ifdef DEBUG
//    std::cout << "MAX Weight : " << max_weight << std::endl;
//    std::cout << "Minimum Weight : " << minimum_weight << std::endl;
#endif

    float weight_avg = .0f;
//    if(max_weight == 0.0)max_weight = 1.0;
    for(Particles::iterator it = _particles_state.begin();
        it != _particles_state.end(); it++){
//        weight_avg += it->w/max_weight;
        weight_avg += it->w;
        it->w = it->w / total_weight;        
    }    
//    if(weight_avg==0)weight_avg=last_weight_avg_;
//    _weight_avg = weight_avg == 0.0 ? last_weight_avg_ : weight_avg/params_.num_particles;

    _weight_avg = weight_avg/params_.num_particles;
}

inline void Localization::arrangeTargetPoints(Points &_target_points){
    for(size_t i = 0; i < _target_points.size(); i++){
        for(size_t j = i+1; j < _target_points.size(); j++){
            if(_target_points[i].x > _target_points[j].x){
                _target_points[i] = _target_points[i] + _target_points[j];
                _target_points[j] = _target_points[i] - _target_points[j];
                _target_points[i] = _target_points[i] - _target_points[j];
            }else if(_target_points[i].x == _target_points[j].x){
                if(_target_points[i].y > _target_points[j].y){
                    _target_points[i] = _target_points[i] + _target_points[j];
                    _target_points[j] = _target_points[i] - _target_points[j];
                    _target_points[i] = _target_points[i] - _target_points[j];
                }
            }
        }
    }
}

std::vector<Points > Localization::sampleCircleCandidates(cv::Mat &_points_map, const Points &_points){
    std::vector<Points > points_group;
    if(_points.size() < 1)return points_group;//return empty points group
    Points collected_points;
    constexpr int remap_x = POINTS_MAP_W >> 1;
    constexpr int remap_y = POINTS_MAP_H >> 2;
    cv::Point remap_origin(remap_x, remap_y);
    cv::Point pres_point = cv::Point(remap_origin.x + _points.front().x, remap_origin.y + _points.front().y);
    cv::Point next_point;
    cv::Point secondary_point(-1,-1);
    cv::Point reference_point = pres_point;
    // float prev_grad=0;
    float prev_len = .0f;

    int count = 1;
//    cv::Mat cek_aj = cv::Mat::zeros(FRAME_SIZE,CV_8UC1);
    while(count < _points.size()){

        next_point = cv::Point(-1,-1);

        if(_points_map.at<uchar>(pres_point.y, pres_point.x) == 0){
            pres_point = cv::Point(remap_origin.x + _points[count].x, remap_origin.y + _points[count].y);
            reference_point = pres_point;
            secondary_point = cv::Point(-1,-1);
            // prev_grad = 0;
            prev_len = .0f;
            ++count;
            continue;
        }

        _points_map.at<uchar>(pres_point.y,pres_point.x) = 0;
//        collected_points.push_back(cv::Point(pres_point.x - remap_origin.x , pres_point.y - remap_origin.y));
        collected_points.emplace_back(cv::Point(pres_point.x, pres_point.y));

        for(size_t i=0;i<radial_pattern_.size();i++){
            int pos_x = pres_point.x + radial_pattern_[i].first;
            int pos_y = pres_point.y + radial_pattern_[i].second;

            if(pos_x < 0 || pos_x >= POINTS_MAP_W || pos_y < 0 || pos_y >= POINTS_MAP_H)continue;

            if(_points_map.at<uchar>(pos_y,pos_x) == 0)continue;

            float diff_x = pos_x - reference_point.x;
            float diff_y = pos_y - reference_point.y;
            // float abs_grad = fabs(diff_y/(diff_x+1e-6));
            float len_to_ref = sqrt(diff_x*diff_x + diff_y*diff_y);
//            cek_aj.at<uchar>(pos_y,pos_x) = 255;
//            cv::imshow("CEK",cek_aj);
//            cv::waitKey(0);
            if(collected_points.size() == 1){//first attempt

                next_point = cv::Point(pos_x,pos_y);

                // prev_grad = abs_grad;
                prev_len = len_to_ref;

            }else {
                if(secondary_point.x == -1){
                    secondary_point = cv::Point(pos_x,pos_y);
                }else if((len_to_ref - prev_len) >= .0f){
                    //if(//(abs_grad-prev_grad) >= -0.1f &&
                    next_point = cv::Point(pos_x,pos_y);
                    // prev_grad = abs_grad;
                    prev_len = len_to_ref;
                }

                break;
            }
        }

        //there are no neighborhood anymore
        if(next_point.x == -1){
            if(secondary_point.x >= 0){
                pres_point = secondary_point;
                // prev_grad = 0;
                prev_len = 0;
                secondary_point = cv::Point(-2, -2);
            }else{
                points_group.push_back(collected_points);
                collected_points.clear();
//                cek_aj = cv::Mat::zeros(FRAME_SIZE,CV_8UC1);
            }
        }else{
           pres_point = next_point;
        }
    }

    return points_group;
}

void Localization::getFeaturesModels(Points &_target_points,
                                 Vectors3 &_line_models,
                                 vision_utils::Features &_features_arg){
    vision_utils::Features _features;
    cv::Mat points_map = cv::Mat::zeros(POINTS_MAP_H, POINTS_MAP_W, CV_8UC1);
    constexpr int remap_x = POINTS_MAP_W >> 1;
    constexpr int remap_y = POINTS_MAP_H >> 2;
    cv::Point remap_origin(remap_x, remap_y);
    cv::Vec4f best_circle(-1.0f, -1.0f, std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    constexpr float CENTER_CIRCLE_RADIUS = (CENTER_CIRCLE_DIAMETER >> 1);

    for(size_t i = 0; i < _target_points.size(); i++){
        points_map.at<uchar>(remap_origin.y + _target_points[i].y, remap_origin.x + _target_points[i].x) = 255;
    }

    std::vector<Points > grouped_points = sampleCircleCandidates(points_map, _target_points);
#ifdef DEBUG
cv::Mat visualize = cv::Mat::zeros(points_map.size(), CV_8UC3);
    for(std::vector<Points>::iterator it=grouped_points.begin();
        it!=grouped_points.end();it++){
        cv::Scalar color = cv::Scalar(rand()%255,rand()%255,rand()%255);
        for(Points::iterator it2=it->begin();
            it2!=it->end();it2++){
            visualize.at<cv::Vec3b >(it2->y,it2->x)[0] = color[0];
            visualize.at<cv::Vec3b >(it2->y,it2->x)[1] = color[1];
            visualize.at<cv::Vec3b >(it2->y,it2->x)[2] = color[2];
        }
    }
#endif

    int most_points=0;
#ifdef DEBUG
    std::cout << "Circle Candidates : " << std::endl;
#endif
    for(size_t i=0;i<grouped_points.size();i++){

//        for(size_t j=0;j<grouped_points[i].size();j++){
//            draw_remain.at<uchar>(grouped_points[i][j].y + remap_origin.y,grouped_points[i][j].x + remap_origin.x) = 255;
//            std::cout << cv::Point(grouped_points[i][j].x + remap_origin.x, grouped_points[i][j].y + remap_origin.y) << std::endl;
//        }
//        cv::imshow("TEST",draw_remain);
//        cv::waitKey(0);
//        arrangeTargetPoints(grouped_points[i]);
        Points sample_circle = grouped_points[i];
        if(sample_circle.size() < 10)continue;
#ifdef DEBUG
        std::cout << "GP Density : " << cv::arcLength(sample_circle,false) / sample_circle.size() << std::endl;
#endif
        // std::cout << "PERIMETER LEN :" << cv::arcLength(grouped_points[i],false) << std::endl;
        cv::Vec4f circle_param = FitCircle::getInstance()->newtonPrattMethod(sample_circle, Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);
//        cv::circle(visualize,cv::Point(circle_param[0],circle_param[1])
//                ,circle_param[2],cv::Scalar(255),2);
//        circle_param[3] /= (circle_param[2]*circle_param[2]);
#ifdef DEBUG
        std::cout << " Group Size : " << (int)sample_circle.size() << " ; Circle Param " << circle_param << std::endl;
#endif
        if(circle_param[3] < best_circle[3] &&
                circle_param[3] > .0f && circle_param[3] < circle_cost &&
                circle_param[2] > .75f * CENTER_CIRCLE_RADIUS &&
                circle_param[2] < 1.25f * CENTER_CIRCLE_RADIUS &&
                sample_circle.size() > most_points){
//            float ctr_dev = sqrt((circle_param[0]-best_circle[0])*(circle_param[0]-best_circle[0]) +
//                    (circle_param[1]-best_circle[1])*(circle_param[1]-best_circle[1]));
//            if(ctr_dev < 10)support++;
//            else support=0

            best_circle = circle_param;
            most_points = sample_circle.size();
        }
    }


    /*Points circle_inliers;
    int best_inliers=0;
    std::cout << "==================================================" << std::endl;
    for(int iter=0;iter<50;iter++){
        Points random_sample;
//        int first_idx = rand()%_target_points.size();
//        int second_idx = std::min((int)_target_points.size()-1, first_idx + 5);
//        int third_idx = std::min((int)_target_points.size()-1, first_idx + 10);
//        int fourth_idx = std::max(0, first_idx - 5);
//        int fifth_idx = std::max(0, first_idx - 10);
//        random_sample.push_back(_target_points[first_idx]);
//        random_sample.push_back(_target_points[second_idx]);
//        random_sample.push_back(_target_points[third_idx]);
//        random_sample.push_back(_target_points[fourth_idx]);
//        random_sample.push_back(_target_points[fifth_idx]);
        random_sample.push_back(_target_points[rand()%_target_points.size()]);
        random_sample.push_back(_target_points[rand()%_target_points.size()]);
        random_sample.push_back(_target_points[rand()%_target_points.size()]);
//        random_sample.push_back(_target_points[rand()%_target_points.size()]);
//        random_sample.push_back(_target_points[rand()%_target_points.size()]);
//        random_sample.push_back(_target_points[rand()%_target_points.size()]);
//        random_sample.push_back(_target_points[rand()%_target_points.size()]);
//        random_sample.push_back(_target_points[rand()%_target_points.size()]);
//        random_sample.push_back(_target_points[rand()%_target_points.size()]);
//        random_sample.push_back(_target_points[rand()%_target_points.size()]);
        cv::Vec4f circle_param = FitCircle::getInstance()->newtonPrattMethod(random_sample, Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);
        std::cout << circle_param << std::endl;
        if(circle_param[2] > 0.9 * (CENTER_CIRCLE_DIAMETER>>1) &&
                circle_param[2] < 1.3 * (CENTER_CIRCLE_DIAMETER>>1) &&
                circle_param[3]/(circle_param[2]*circle_param[2]) < 2){

            int total_inliers=0;

//            for(cv::Point &pt:_target_points){
            Points inliers;
            for(size_t i=0;i<_target_points.size();i++){
                float diff_x = circle_param[0] - _target_points[i].x;
                float diff_y = circle_param[1] - _target_points[i].y;
                float err_to_circle = fabs(sqrt(diff_x*diff_x + diff_y*diff_y) - (CENTER_CIRCLE_DIAMETER>>1));
                if(err_to_circle < 5){
                    total_inliers++;
                    inliers.push_back(_target_points[i]);
                }
            }
            std::cout << total_inliers << std::endl;
            if(total_inliers > best_inliers &&
                    total_inliers > 20){
//                circle_set.push_back(circle_param);
                best_inliers=total_inliers;
                circle_inliers = inliers;
                best_circle=circle_param;
            }
        }
    }
    best_circle = FitCircle::getInstance()->newtonPrattMethod(circle_inliers, Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);

    for(size_t idx=0;idx<circle_inliers_idx.size();idx++){
        _target_points.erase(_target_points.begin() + circle_inliers_idx[idx] - idx);
    }*/

    //remove circle inliers
    if(best_circle[0] >= .0f){
        Points updated_points;
        constexpr float MAX_ERR_TO_CIRCLE = (CENTER_CIRCLE_DIAMETER >> 4);
        Points circle_inliers;
        for(cv::Point &pt:_target_points){
            float diff_x = best_circle[0] - remap_origin.x - pt.x;
            float diff_y = best_circle[1] - remap_origin.y - pt.y;
            float err_to_circle = fabs(sqrt(diff_x*diff_x + diff_y*diff_y) - CENTER_CIRCLE_RADIUS);
            if(err_to_circle < MAX_ERR_TO_CIRCLE){
                circle_inliers.push_back(pt);
                continue;
            }
//            draw_remain.at<uchar>(remap_origin.y + pt.y, remap_origin.x + pt.x) = 255;
            updated_points.push_back(pt);
        }
        best_circle[0] = -1.0f;
        best_circle[1] = -1.0f;
        best_circle[2] = -1.0f;
        best_circle[3] = -1.0f;
//        std::cout << "Circle Inliers Count : " << circle_inliers.size() << std::endl;
        if(circle_inliers.size() > MIN_CIRCLE_INLIERS){
            cv::Vec4f circle_param = FitCircle::getInstance()->newtonPrattMethod(circle_inliers, Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);
#ifdef DEBUG
            std::cout << "LAST Circle Param : " << circle_param << std::endl;
#endif
            if(circle_param[3] > .0f && circle_param[3] < circle_cost &&
                circle_param[2] > .75f * CENTER_CIRCLE_RADIUS &&
                circle_param[2] < 1.25f * CENTER_CIRCLE_RADIUS){
                    best_circle[0] = circle_param[0] + remap_origin.x;
                    best_circle[1] = circle_param[1] + remap_origin.y;
                    best_circle[2] = circle_param[2];
                    best_circle[3] = circle_param[3];
                    _target_points=updated_points;
            }
        }
//        if( most_point <  !(inliers_count > (most_points << 1) ) )best_circle[0] = -1;
    }
    std::vector<Points > segline_inliers;
// For Grouped Points
//    for(size_t i=0;i<grouped_points.size();i++){
//        Points inliers;
//        geometry_msgs::Vector3 model;

//        if(grouped_points[i].size() < 6)continue;

//        //Get line models & Update target points
//        LocalizationUtils::getInstance()->RANSAC(grouped_points[i], model, 3, 20, inlier_error, grouped_points[i].size()>>1, inliers);
//        std::cout << "Model : " << model.x << " ; " << model.y << std::endl;
//        //Unable to get line model
//        if(model.x == 0 && model.y == 0)
//            continue;

//        vision_utils::Feature feature_data;
//        feature_data.param1 = inliers[0].x;
//        feature_data.param2 = model.x + feature_data.param1*model.y;
//        feature_data.param3 = inliers[inliers.size()-1].x;
//        feature_data.param4 = model.x + feature_data.param3*model.y;
//        float orientation = atan2(feature_data.param4-feature_data.param2, feature_data.param3-feature_data.param1)*Math::RAD2DEG;
//        feature_data.orientation = orientation < 0 ? 180+orientation:orientation;
//        feature_data.feature_type = 4;
//        _features.feature.push_back(feature_data);

//        _line_models.push_back(model);

//        segline_inliers.push_back(inliers);
//#ifdef DEBUG
//        cv::line(visualize,cv::Point(feature_data.param1 + remap_origin.x, feature_data.param2 + remap_origin.y),
//                 cv::Point(feature_data.param3 + remap_origin.x, feature_data.param4 + remap_origin.y),
//                 cv::Scalar(255,0,255),2);
//#endif
//    }


    for(int i = MAX_LINE_MODEL; i--;){
        Points inliers;
        geometry_msgs::Vector3 model;

        //Unsufficent data
        if(_target_points.size() < MIN_LINE_INLIERS)
            break;

        //Get line models & Update target points
        LocalizationUtils::getInstance()->RANSAC(_target_points, model, Alfarobi::RANSAC_NUM_SAMPLES, Alfarobi::RANSAC_MAX_ITER, inlier_error, MIN_LINE_INLIERS, inliers);

        //Unable to get line model
        if(model.x == 0 && model.y == 0)
            continue;

        vision_utils::Feature feature_data;
        feature_data.param1 = inliers.front().x;
        feature_data.param2 = model.x + feature_data.param1*model.y;
        feature_data.param3 = inliers.back().x;
        feature_data.param4 = model.x + feature_data.param3*model.y;
        int diff_x = feature_data.param1 - feature_data.param3;
        int diff_y = feature_data.param2 - feature_data.param4;
        if(sqrt(diff_x*diff_x + diff_y*diff_y) < MIN_LINE_LENGTH){
            continue;
        }
        float orientation = atan2(fabs(feature_data.param3-feature_data.param1),
                                 fabs(feature_data.param4-feature_data.param2)) * Math::RAD2DEG;

//        std::cout << orientation << std::endl;
        feature_data.orientation = orientation; //> 90 ? orientation - 90:orientation;//orientation < 0 ? 180 + orientation:orientation;
        feature_data.feature_type = 4;
        _features.feature.push_back(feature_data);

        _line_models.push_back(model);

        segline_inliers.push_back(inliers);
#ifdef DEBUG
        cv::line(visualize,cv::Point(remap_origin.x + feature_data.param1, remap_origin.y + feature_data.param2),
                cv::Point(remap_origin.x + feature_data.param3, remap_origin.y + feature_data.param4),
                cv::Scalar(255,0,255), 1);
#endif

    }

    //merge smiliar line
    // std::cout << "==================" << std::endl;
    for(size_t i = 0; i < _line_models.size(); i++){
        for(size_t j = i+1; j < _line_models.size(); j++){
            float grad_ratio = fabs(_line_models[i].y) > fabs(_line_models[j].y) ?
                        fabs(_line_models[j].y/_line_models[i].y) : fabs(_line_models[i].y/_line_models[j].y);
            bool dominant = segline_inliers[i].size() >= segline_inliers[j].size();
            int idx1 = dominant ? i : j;
            int idx2 = dominant ? j : i;
            float tip1_dist = fabs(_line_models[idx1].y*_features.feature[idx2].param1 - _features.feature[idx2].param2 + _line_models[idx1].x)/sqrt(_line_models[idx1].y*_line_models[idx1].y + 1);
            float tip2_dist = fabs(_line_models[idx1].y*_features.feature[idx2].param3 - _features.feature[idx2].param4 + _line_models[idx1].x)/sqrt(_line_models[idx1].y*_line_models[idx1].y + 1);
            //  std::cout << "Grad Ratio : " << grad_ratio << std::endl;
            //  std::cout << "Bias Diff : " << tip1_dist + tip2_dist << std::endl;
            if(grad_ratio > .4f &&
                    (tip1_dist + tip2_dist) < 50.0f){
//                bool dominant = segline_inliers[i].size() > segline_inliers[j].size();
                // _line_models[i].y = (_line_models[i].y + _line_models[j].y)/2;
                _line_models[i] = dominant ? _line_models[i] : _line_models[j];
                _features.feature[i].param1 = std::min(_features.feature[i].param1,_features.feature[j].param1);
                _features.feature[i].param2 = _line_models[i].x + _line_models[i].y*_features.feature[i].param1;
                _features.feature[i].param3 = std::max(_features.feature[i].param3,_features.feature[j].param3);
                _features.feature[i].param4 = _line_models[i].x + _line_models[i].y*_features.feature[i].param3;
                float orientation = atan2(fabs(_features.feature[i].param3-_features.feature[i].param1),
                                         fabs(_features.feature[i].param4-_features.feature[i].param2))*Math::RAD2DEG;
                _features.feature[i].orientation = orientation; //> 90 ? orientation - 90:orientation;//orientation < 0 ? 180+orientation:orientation;
                _features.feature.erase(_features.feature.begin() + j);
                _line_models.erase(_line_models.begin() + j);
                segline_inliers[i].insert(segline_inliers[i].end(),segline_inliers[j].begin(),segline_inliers[j].end());
                segline_inliers.erase(segline_inliers.begin() + j);
                --j;
            }
        }
    }

    //remove line with circle inliers
#ifdef DEBUG
    //   std::cout << "Line with circle inliers : " << std::endl;
#endif
    for(int i = 0; i < (int)_line_models.size(); i++){
        int diff_x = _features.feature[i].param1 - _features.feature[i].param3;
        int diff_y = _features.feature[i].param2 - _features.feature[i].param4;
        if(sqrt(diff_x*diff_x + diff_y*diff_y) < MIN_LINE_LENGTH){
            _features.feature.erase(_features.feature.begin() + i);
            _line_models.erase(_line_models.begin() + i);
            segline_inliers.erase(segline_inliers.begin() + i);
            --i;
            continue;
        }
        cv::Vec4f circle_param = FitCircle::getInstance()->newtonPrattMethod(segline_inliers[i], Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);
//        circle_param[3] /= (circle_param[2]*circle_param[2]);
#ifdef DEBUG
        //   std::cout << "Circle Param : " << circle_param << std::endl;
#endif
        if(circle_param[3] < 15.0f && //Approximate it !!!
//           circle_param[2] > 0.7 * (CENTER_CIRCLE_DIAMETER>>1) &&
           circle_param[2] < CENTER_CIRCLE_DIAMETER){
            _line_models.erase(_line_models.begin() + i);
            _features.feature.erase(_features.feature.begin() + i);
            --i;
        }
    }

    
    if(best_circle[0] != -1){        
        // std::cout << best_circle << std::endl;
        vision_utils::Feature feature_data;
        feature_data.param1 = best_circle[0] - remap_origin.x;
        feature_data.param2 = best_circle[1] - remap_origin.y;
        feature_data.param3 = best_circle[2];
        float diff_x = feature_data.param1;
        float diff_y = feature_data.param2;
        feature_data.param4 = sqrt(diff_x*diff_x + diff_y*diff_y);
        feature_data.orientation = atan2(diff_x,diff_y) * Math::RAD2DEG;
        //  std::cout << "Cirlce Orientation : " << feature_data.orientation << std::endl;
        feature_data.feature_type = 3;
        
#ifdef DEBUG
        cv::circle(visualize,cv::Point(best_circle[0], best_circle[1])
                ,best_circle[2],cv::Scalar(255),1);
#endif

        //check line collision with circle
        constexpr float MAX_DIST_TO_CIRCLE = (CENTER_CIRCLE_DIAMETER >> 4);
        for(int i=0;i<(int)_line_models.size();i++){
            geometry_msgs::Vector3 model = _line_models[i];
            float check_collision = fabs((model.y * feature_data.param1) - feature_data.param2 + model.x)/sqrt(model.y*model.y+1);
            if(!(check_collision > feature_data.param3 + MAX_DIST_TO_CIRCLE ||
                 check_collision < feature_data.param3 - MAX_DIST_TO_CIRCLE)){
                _features.feature.erase(_features.feature.begin() + i);
                _line_models.erase(_line_models.begin() + i);
                --i;
            }
        }

        _features.feature.push_back(feature_data);
    }

    if(_features.feature.size() > 0){
        features_present_ = best_circle[0] != -1 ? 999 : _features.feature.size();
        _features_arg.feature.clear();
        _features_arg.feature = _features.feature;
        lost_features_ = false;
    }

//    for(size_t i=0;i<_line_models.size();i++){
//        std::cout << "Orientation : " << _features.feature[i].orientation << std::endl;

//    }
#ifdef DEBUG
    if(debug_viz_mode==2)debug_viz_ = visualize.clone();
//    cv::imshow("VIZ",visualize);
//    cv::waitKey(0);
    // visualize = cv::Mat::zeros(FRAME_SIZE,CV_8UC1);
    // for(vision_utils::Feature &feature_data:_features.feature){
    //     cv::line(visualize,cv::Point(feature_data.param1+remap_origin.x, feature_data.param2+remap_origin.y),
    //             cv::Point(feature_data.param3+remap_origin.x, feature_data.param4+remap_origin.y),
    //             cv::Scalar(255,0,255),2);
    // }
//   cv::imshow("VIZ2",visualize);
//   cv::waitKey(0);
#endif
}
//overloading function for METHOD 2
#ifndef METHOD_1
void Localization::getFeaturesModels(Points &_target_points,
                                 Vectors3 &_line_models,
                                 vision_utils::Features &_features, vision_utils::LineTip &_line_tip){

    cv::Mat points_map = cv::Mat::zeros(FRAME_HEIGHT,FRAME_WIDTH, CV_8UC1);
    cv::Mat visualize = cv::Mat::zeros(FRAME_HEIGHT,FRAME_WIDTH, CV_8UC3);
    cv::Point remap_origin(FRAME_WIDTH >> 1,0);

    std::vector<Points > segline_inliers;

    Points target_pt;
    Vectors3 original_line_models;
//    std::cout << "=============" << std::endl;
    for(int i = MAX_LINE_MODEL; i--;){
        Points inliers;
        geometry_msgs::Vector3 model;
//        std::cout << _target_points.size() << std::endl;
        //Unsufficent data
        if(_target_points.size() < MIN_LINE_INLIERS)
            break;

        //Get line models & Update target points
        LocalizationUtils::getInstance()->RANSAC(_target_points, model, Alfarobi::RANSAC_NUM_SAMPLES, Alfarobi::RANSAC_MAX_ITER, inlier_error, MIN_LINE_INLIERS, inliers);
//        std::cout << model.x << " ; " << model.y << std::endl;
        //Unable to get line model
        if(model.x == 0 || model.y == 0)
            continue;
        
        original_line_models.push_back(model);
        cv::Point tip1(inliers.front().x,model.x + inliers.front().x*model.y);
        cv::Point tip2(inliers.back().x,model.x + inliers.back().x*model.y);
        target_pt.push_back(tip1);
        target_pt.push_back(tip2);
        
        segline_inliers.push_back(inliers);

        geometry_msgs::Point tip_1,tip_2;
        tip_1.x = tip1.x;
        tip_1.y = tip1.y;
        tip_2.x = tip2.x;
        tip_2.y = tip2.y;
        _line_tip.tip1.push_back(tip_1);
        _line_tip.tip2.push_back(tip_2);    
#ifdef DEBUG
        cv::line(points_map,tip1,tip2,cv::Scalar(100),2);
#endif
    }

    Points projected_points = pointsProjection(target_pt);
    target_pt.clear();
    for(size_t i=0;i<projected_points.size()/2;i++){
        int idx1 = 2*i;
        int idx2 = 2*i+1;        

        vision_utils::Feature feature_data;
        feature_data.param1 = projected_points[idx1].x;
        feature_data.param2 = projected_points[idx1].y;
        feature_data.param3 = projected_points[idx2].x;
        feature_data.param4 = projected_points[idx2].y;
        float orientation = atan2(feature_data.param4-feature_data.param2, feature_data.param3-feature_data.param1)*Math::RAD2DEG;
        feature_data.orientation = orientation < 0 ? 180+orientation:orientation;
        feature_data.feature_type = 4;
        _features.feature.push_back(feature_data);
        
        geometry_msgs::Vector3 line_model;
        line_model.y = (projected_points[idx2].y - projected_points[idx1].y)/(projected_points[idx2].x - projected_points[idx1].x + 1e-6);
        line_model.x = projected_points[idx1].y - line_model.y*projected_points[idx1].x;
        _line_models.push_back(line_model);  
        
#ifdef DEBUG
        cv::line(visualize,cv::Point(remap_origin.x + feature_data.param1, remap_origin.y + feature_data.param2),
                cv::Point(remap_origin.x + feature_data.param3, remap_origin.y + feature_data.param4),
                cv::Scalar(255,0,255),2);
#endif
    }
    
    //merge smiliar line
//     std::cout << "==================" << std::endl;
    for(size_t i=0;i<_line_models.size();i++){
        for(size_t j=i+1;j<_line_models.size();j++){
            float grad_ratio = fabs(_line_models[i].y) > fabs(_line_models[j].y) ?
                        fabs(_line_models[j].y/_line_models[i].y) : fabs(_line_models[i].y/_line_models[j].y);
//             std::cout << "Grad Ratio : " << grad_ratio << std::endl;
             bool dominant = segline_inliers[i].size() >= segline_inliers[j].size();
             int idx1 = dominant ? i : j;
             int idx2 = dominant ? j : i;
             float tip1_dist = fabs(_line_models[idx1].y*_features.feature[idx2].param1 - _features.feature[idx2].param2 + _line_models[idx1].x)/sqrt(_line_models[idx1].y*_line_models[idx1].y + 1);
             float tip2_dist = fabs(_line_models[idx1].y*_features.feature[idx2].param3 - _features.feature[idx2].param4 + _line_models[idx1].x)/sqrt(_line_models[idx1].y*_line_models[idx1].y + 1);
//             std::cout << "Bias Diff : " << tip1_dist + tip2_dist << std::endl;
            if(grad_ratio > 0.35 &&
                    (tip1_dist + tip2_dist) < 50){
                // _line_models[i].y = (_line_models[i].y + _line_models[j].y)/2;
                _line_models[i] = dominant ? _line_models[i] : _line_models[j];
                _features.feature[i].param1 = std::min(_features.feature[i].param1,_features.feature[j].param1);
                _features.feature[i].param2 = _line_models[i].x + _line_models[i].y*_features.feature[i].param1;
                _features.feature[i].param3 = std::max(_features.feature[i].param3,_features.feature[j].param3);
                _features.feature[i].param4 = _line_models[i].x + _line_models[i].y*_features.feature[i].param3;                                

                float orientation = atan2(_features.feature[i].param4-_features.feature[i].param2,
                                          _features.feature[i].param3-_features.feature[i].param1)*Math::RAD2DEG;
                _features.feature[i].orientation = orientation < 0 ? 180+orientation:orientation;
                _features.feature.erase(_features.feature.begin() + j);

                _line_models.erase(_line_models.begin() + j);
                
                original_line_models[i] = dominant ? original_line_models[i] : original_line_models[j];
                _line_tip.tip1[i].x = std::min(_line_tip.tip1[i].x,_line_tip.tip1[j].x);
                _line_tip.tip1[i].y = original_line_models[i].x + original_line_models[i].y * _line_tip.tip1[i].x;
                _line_tip.tip2[i].x = std::max(_line_tip.tip2[i].x,_line_tip.tip2[j].x);
                _line_tip.tip2[i].y = original_line_models[i].x + original_line_models[i].y * _line_tip.tip2[i].x;

                _line_tip.tip1.erase(_line_tip.tip1.begin() + j);
                _line_tip.tip2.erase(_line_tip.tip2.begin() + j);

                original_line_models.erase(original_line_models.begin() + j);

                segline_inliers[i].insert(segline_inliers[i].end(),segline_inliers[j].begin(),segline_inliers[j].end());
                segline_inliers.erase(segline_inliers.begin() + j);

                j--;
            }
        }
    }
    
    for(int i=0;i<(int)_line_models.size();i++){
        int diff_x = _features.feature[i].param1 - _features.feature[i].param3;
        int diff_y = _features.feature[i].param2 - _features.feature[i].param4;
        if(sqrt(diff_x*diff_x + diff_y*diff_y) < MIN_LINE_LENGTH){
            _features.feature.erase(_features.feature.begin() + i);
            _line_models.erase(_line_models.begin() + i);
            segline_inliers.erase(segline_inliers.begin() + i);
            i--;
            continue;
        }
    }
    
    features_present_ = _features.feature.size();
#ifdef DEBUG
    if(debug_viz_mode==2)debug_viz_ = visualize.clone();
//  cv::imshow("VIZ",visualize);
//  cv::imshow("PM",points_map);
#endif
}
#endif
int Localization::getIntersectionType(const cv::Mat &_segmented_image, const cv::Vec2f _model[2],const cv::Point &_center){
    int sampling_radius = 30;
    int min_length=std::numeric_limits<int>::max();
    int segment_length[4] = {0, 0, 0, 0};
    for(int i=_center.x;i<_center.x+sampling_radius && i <_segmented_image.cols;i++){
        int j = _model[0][0] + _model[0][1]*i;
        if(_segmented_image.at<uchar > (j,i)>0){
            ++segment_length[0];
        }
    }

    if(segment_length[0] < sampling_radius){
        min_length = segment_length[0];
    }

    for(int i=_center.x;i > _center.x-sampling_radius && i > 0;i--){
        int j = _model[0][0] + _model[0][1]*i;
        if(_segmented_image.at<uchar > (j,i)>0){
            ++segment_length[1];
        }
    }

    if(segment_length[1] < min_length){
        min_length=segment_length[1];
    }

    for(int i=_center.x;i<_center.x+sampling_radius && i <_segmented_image.cols;i++){
        int j = _model[1][0] + _model[1][1]*i;
        if(_segmented_image.at<uchar > (j,i)>0){
            ++segment_length[2];
        }
    }

    if(segment_length[2] < min_length){
        min_length=segment_length[2];
    }

    for(int i=_center.x;i > _center.x-sampling_radius && i > 0;i--){
        int j = _model[1][0] + _model[1][1]*i;
        if(_segmented_image.at<uchar > (j,i)>0){
            ++segment_length[3];
        }
    }

    if(segment_length[3] < min_length){
        min_length=segment_length[3];
    }

    int total = segment_length[0] + segment_length[1] + segment_length[2] + segment_length[3];
    if(total>=sampling_radius*4){
        return 4;
    }else if(total>=sampling_radius*3){
        return 3;
    }else if(total>=sampling_radius*2){
        if(((segment_length[0] < sampling_radius && segment_length[2] < sampling_radius) && (segment_length[1] >= sampling_radius && segment_length[3] >= sampling_radius)) ||
                ((segment_length[0] >= sampling_radius && segment_length[2] >= sampling_radius) && (segment_length[1] < sampling_radius && segment_length[3] < sampling_radius)))
            return 2;
        else
            return 1;
    }else {
        return 0;
    }
}

vision_utils::Features Localization::getLineIntersection(Vectors3 _line_models){
    // Not used yet
    vision_utils::Features features;

    for(size_t i=0; i < _line_models.size(); i++){
        for(size_t j = i+1; j < _line_models.size(); j++){
            // Check gradient difference
            if(fabs(_line_models[i].y - _line_models[j].y) < 0.05)
                continue;

            // Solve x for Ax = b
            cv::Mat A = cv::Mat::ones(2, 2, CV_32FC1);
            A.at<float > (0) = -_line_models[i].y;
            A.at<float > (2) = -_line_models[j].y;
            cv::Mat b = (cv::Mat_<float >(2,1) << _line_models[i].x, _line_models[j].x);
            cv::Mat x = A.inv()*b;
//            cv::Vec2f intersect_line[2] = {_line_models[i], _line_models[j]};
//            int features_type = getIntersectionType(segmented_white_, intersect_line, cv::Point(x.at<float > (0), x.at<float > (1)));
            int features_type = 2;
            cv::Vec6f temp;
            temp[0] = x.at<float>(0);
            temp[1] = x.at<float>(1);
            temp[5] = features_type - 2;
            // image_in_ not used
            switch(features_type){
                case 2:
                putText(image_in_, "L", cv::Point(x.at<float>(0), x.at<float>(1)), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(25,25,0), 2);
                circle(image_in_, cv::Point(x.at<float>(0), x.at<float>(1)), 4, cv::Scalar(0,255,0), CV_FILLED);
//                features.push_back(line_feature);
                break;
                case 3:
                putText(image_in_, "T", cv::Point(x.at<float>(0), x.at<float>(1)), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(25,25,0), 2);
                circle(image_in_, cv::Point(x.at<float>(0), x.at<float>(1)), 4, cv::Scalar(0,255,0), CV_FILLED);
//                features.push_back(line_feature);
                break;
                case 4:
                putText(image_in_, "X", cv::Point(x.at<float>(0), x.at<float>(1)), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(25,25,0), 2);
                circle(image_in_, cv::Point(x.at<float>(0), x.at<float>(1)), 4, cv::Scalar(0,255,0), CV_FILLED);
//                features.push_back(line_feature);
                break;
                default:
                circle(image_in_, cv::Point(x.at<float>(0), x.at<float>(1)),4, cv::Scalar(0,255,0), CV_FILLED);
                break;
            }
        }
    }
    return features;
}

Points Localization::pointsProjection(const Points &_points, bool ball){
    Points projected_point;
    cv::Mat points_map = cv::Mat::zeros(POINTS_MAP_H, POINTS_MAP_W, CV_8UC1);
#ifdef DEBUG
    cv::Mat target_points_viz = cv::Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
    cv::Mat unrotated_pm = points_map.clone();
    cv::Mat compensated = target_points_viz.clone();
    cv::Mat distorted_tp = target_points_viz.clone();
#endif
    constexpr int remap_x = POINTS_MAP_W >> 1;
    constexpr int remap_y = POINTS_MAP_H >> 2;

    constexpr float ctr_frame_x = FRAME_WIDTH >> 1;
    constexpr float ctr_frame_y = FRAME_HEIGHT >> 1;
    cv::Point2f center_frame(ctr_frame_x, ctr_frame_y);
    float K1 = camera_info_.D[0];
    float K2 = camera_info_.D[1];
    float K3 = camera_info_.D[4];
    float P1 = camera_info_.D[2];
    float P2 = camera_info_.D[3];
    float cx = camera_info_.K[2];
    float cy = camera_info_.K[5];
    float fx = camera_info_.K[0];
    float fy = camera_info_.K[4];
    float ctr_x = (center_frame.x - cx)/fx;
    float ctr_y = (center_frame.y - cy)/fy;

//    const float PI_2 = CV_PI/2;
    forwardKinematics();
    float c_roll_comp = cos(CAMERA_ORIENTATION.coeff(0));
    float s_roll_comp = sin(CAMERA_ORIENTATION.coeff(0));

    float c_pan_servo = cos(pan_servo_angle_ + pan_rot_comp_);//cos(CAMERA_ORIENTATION.coeff(2));
    float s_pan_servo = sin(pan_servo_angle_ + pan_rot_comp_);//sin(CAMERA_ORIENTATION.coeff(2));

    float shift_pixel = (shift_const_ + ctr_frame_y)*(1 - c_roll_comp);
    for(Points::const_iterator it=_points.begin();
        it != _points.end(); it++){        

        float xn = (it->x - cx)/fx;
        float yn = (it->y - cy)/fy;
        float diff_x = xn - ctr_x;
        float diff_y = yn - ctr_y;
        float rd_2 = diff_x*diff_x + diff_y*diff_y;
        float rd_4 = rd_2*rd_2;
        float rd_6 = rd_4*rd_2;
        float radial_distort = (1.0f + K1*rd_2 + K2*rd_4 + K3*rd_6);

        float undistort_x = xn*(radial_distort) +
                2.0f*P1*xn*yn + P2*(rd_2 + 2.0f*xn*xn);

        float undistort_y = yn*(radial_distort) +
                P1*(rd_2 + 2.0f*yn*yn) + 2.0f*P2*xn*yn;

        undistort_x = fx*undistort_x + cx;
        undistort_y = fy*undistort_y + cy;

        float trans_x = undistort_x - center_frame.x;
        float trans_y = undistort_y - center_frame.y;

    //   float trans_x = it->x - center_frame.x;
    //   float trans_y = it->y - center_frame.y;

        //projection start here

//        float compensated_x = center_frame.x + trans_x*cos(roll_compensation_) + trans_y*sin(roll_compensation_);
//        float compensated_y = center_frame.y - trans_x*sin(roll_compensation_) + trans_y*cos(roll_compensation_);

//        float roll_comp = (1 - fabs(pan_servo_angle_)/PI_2)*roll_compensation_ + (pan_servo_angle_/PI_2) * offset_head_;
//        float roll_comp = cos(pan_servo_angle_)*roll_compensation_ + sin(pan_servo_angle_)*offset_head_;

        float compensated_x = center_frame.x + trans_x*c_roll_comp + trans_y*s_roll_comp + shift_pixel;
        float compensated_y = center_frame.y - trans_x*s_roll_comp + trans_y*c_roll_comp + shift_pixel;

//        float offset_pan = panAngleDeviation(undistort_x);
//        float offset_tilt = tiltAngleDeviation(undistort_y);

        float offset_pan = panAngleDeviation(compensated_x);
        float offset_tilt = tiltAngleDeviation(compensated_y);

//        float distance_y = verticalDistance(offset_tilt);
        float distance_y = verticalDistance(offset_tilt);
        float distance_x = horizontalDistance(distance_y, offset_pan);
        // std::cout << "OP : " << offset_pan << std::endl;
        // std::cout << " X : " << distance_x << " ; Y : " << distance_y << std::endl;

        if(sqrt(distance_x*distance_x + distance_y*distance_y) < 55.0 && !ball)//ignore feature in less than 55 cm
            continue;

        float rotated_x = distance_x*c_pan_servo + distance_y*s_pan_servo;
        float rotated_y = -distance_x*s_pan_servo + distance_y*c_pan_servo;

        // Robot local coordinate in real world

        cv::Point local_coord(rotated_x, rotated_y);

        int mapped_x = remap_x + local_coord.x;
        int mapped_y = remap_y + local_coord.y;
#ifdef DEBUG
        target_points_viz.at<uchar>(it->y,it->x) = 255;
        if(compensated_x > 0 && compensated_x < target_points_viz.cols &&
           compensated_y > 0 && compensated_y < target_points_viz.rows &&
           debug_viz_mode == 4)
            compensated.at<uchar>(compensated_y,compensated_x) = 255;
#endif
        if(mapped_x < 0 || mapped_x >= POINTS_MAP_W ||
           mapped_y < 0 || mapped_y >= POINTS_MAP_H ||
           points_map.at<uchar>(mapped_y, mapped_x) > 0){
            continue;
        }

#ifdef DEBUG
        int map_unrotate_x = remap_x + distance_x;
        int map_unrotate_y = remap_y + distance_y;
        if(map_unrotate_x > 0 && map_unrotate_x < POINTS_MAP_W &&
           map_unrotate_y > 0 && map_unrotate_y < POINTS_MAP_H &&
           debug_viz_mode == 3)
            unrotated_pm.at<uchar>(map_unrotate_y, map_unrotate_x) = 255;
            
        if(undistort_x > 0 && undistort_x < target_points_viz.cols &&
           undistort_y > 0 && undistort_y < target_points_viz.rows &&
           debug_viz_mode == 5)
            distorted_tp.at<uchar>(undistort_y,undistort_x) = 255;
#endif

        projected_point.push_back(local_coord);
        points_map.at<uchar>(mapped_y, mapped_x) = 255;
        
//        cv::circle(draw,cv::Point(,),1,cv::Scalar(255),CV_FILLED);
    }
#ifdef DEBUG
    if(!ball){
        switch(debug_viz_mode){
            case 0:debug_viz_ = target_points_viz.clone();break;
            case 1:debug_viz_ = points_map.clone();break;
            case 3:debug_viz_ = unrotated_pm.clone();break;
            case 4:debug_viz_ = compensated.clone();break;
            case 5:debug_viz_ = distorted_tp.clone();break;
            default:break;
        }
    }
#endif
//     cv::imshow("Projected Points",points_map);
//     cv::imshow("Target Points",target_points_viz);


    return projected_point;
}

Points Localization::fastPointsProjection(const Points &_points, bool ball){
    Points projected_point;
    cv::Mat points_map = cv::Mat::zeros(POINTS_MAP_H, POINTS_MAP_W, CV_8UC1);
#ifdef DEBUG
    cv::Mat target_points_viz = cv::Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH,CV_8UC1);
#endif
    constexpr int remap_x = POINTS_MAP_W >> 1;
    constexpr int remap_y = POINTS_MAP_H >> 2;

    constexpr float ctr_frame_x = FRAME_WIDTH >> 1;
    constexpr float ctr_frame_y = FRAME_HEIGHT >> 1;
    cv::Point2f center_frame(ctr_frame_x, ctr_frame_y);

    __m256 K1 = _mm256_set1_ps(camera_info_.D[0]);
    __m256 K2 = _mm256_set1_ps(camera_info_.D[1]);
    __m256 K3 = _mm256_set1_ps(camera_info_.D[4]);
    __m256 P1 = _mm256_set1_ps(camera_info_.D[2]);
    __m256 P2 = _mm256_set1_ps(camera_info_.D[3]);

    float s_cx = camera_info_.K[2];
    __m256 cx = _mm256_set1_ps(s_cx);
    float s_cy = camera_info_.K[5];
    __m256 cy = _mm256_set1_ps(s_cy);
    float s_fx = camera_info_.K[0];
    __m256 fx = _mm256_set1_ps(s_fx);
    float s_fy = camera_info_.K[4];
    __m256 fy = _mm256_set1_ps(s_fy);
    float s_ctr_x = (center_frame.x - s_cx)/s_fx;
    __m256 ctr_x = _mm256_set1_ps(s_ctr_x);
    float s_ctr_y = (center_frame.y - s_cy)/s_fy;
    __m256 ctr_y = _mm256_set1_ps(s_ctr_y);

//    __m256 const_one = _mm256_set1_ps(1.0);
//    const float PI_2 = CV_PI/2;
    forwardKinematics();
    float c_roll_comp = cos(CAMERA_ORIENTATION.coeff(0));
    float s_roll_comp = sin(CAMERA_ORIENTATION.coeff(0));

    float c_pan_servo = cos(pan_servo_angle_ + pan_rot_comp_);//cos(CAMERA_ORIENTATION.coeff(2));
    float s_pan_servo = sin(pan_servo_angle_ + pan_rot_comp_);//sin(CAMERA_ORIENTATION.coeff(2));
//    int new_size = (int)_points.size() - (int)_points.size()%8;
    int curr_idx = 0;
    int sz_bound = 0;

    for(int i = 0; i < (int)_points.size(); i+=8){
       auto mem1 = boost::alignment::aligned_alloc(32, 64 * sizeof(float));
       float* mem1_specific = new(mem1) float;
       auto mem2 = boost::alignment::aligned_alloc(32, 64 * sizeof(float));
       float* mem2_specific = new(mem2) float;
       std::unique_ptr<float[], boost::alignment::aligned_delete > arr_x(mem1_specific);
       std::unique_ptr<float[], boost::alignment::aligned_delete > arr_y(mem2_specific);

        for(int j = 0; j < 8; j++){
            curr_idx = i + j;
            sz_bound = (int)(curr_idx < _points.size());
            curr_idx *= sz_bound;
            arr_x.get()[j] = sz_bound * _points[curr_idx].x;
            arr_y.get()[j] = sz_bound * _points[curr_idx].y;
//            std::cout << cv::Point(arr_x[j], arr_y[j]) << std::endl;
        }
        __m256 xn = _mm256_load_ps(arr_x.get());
        __m256 yn = _mm256_load_ps(arr_y.get());
//        __m256 xn = _mm256_set_ps(x1,x2,x3,x4,x5,x6,x7,x8);
//        __m256 yn = _mm256_set_ps(y1,y2,y3,y4,y5,y6,y7,y8);

        __m256 x_norm = _mm256_div_ps(_mm256_sub_ps(xn, cx), fx);
        __m256 y_norm = _mm256_div_ps(_mm256_sub_ps(yn, cy), fy);

        __m256 diff_x = _mm256_sub_ps(x_norm, ctr_x);
        __m256 diff_y = _mm256_sub_ps(y_norm, ctr_y);

        __m256 rd_2 = _mm256_add_ps(_mm256_mul_ps(diff_x, diff_x), _mm256_mul_ps(diff_y, diff_y));
        __m256 rd_4 = _mm256_mul_ps(rd_2, rd_2);
        __m256 rd_6 = _mm256_mul_ps(rd_4, rd_2);

        __m256 radial_distort = _mm256_add_ps(_mm256_add_ps(
                                _mm256_add_ps(_mm256_set1_ps(1.0f),
                                _mm256_mul_ps(K1, rd_2)),
                                _mm256_mul_ps(K2, rd_4)),
                                _mm256_mul_ps(K3, rd_6));

        __m256 xy_norm = _mm256_mul_ps(x_norm, y_norm);

        __m256 xterm1 = _mm256_mul_ps(x_norm, radial_distort);
        __m256 xterm2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), P1), xy_norm);
        __m256 xterm3 = _mm256_mul_ps(P2, _mm256_add_ps(rd_2, _mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(x_norm, x_norm))));

        __m256 new_x = _mm256_add_ps(xterm1, _mm256_add_ps(xterm2, xterm3));

        __m256 yterm1 = _mm256_mul_ps(y_norm, radial_distort);
        __m256 yterm2 = _mm256_mul_ps(P1, _mm256_add_ps(rd_2, _mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(y_norm, y_norm))));
        __m256 yterm3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), P2), xy_norm);

        __m256 new_y = _mm256_add_ps(yterm1, _mm256_add_ps(yterm2, yterm3));

        __m256 undistort_x = _mm256_fmadd_ps(fx, new_x, cx);
        __m256 undistort_y = _mm256_fmadd_ps(fy, new_y, cy);

        __m256 trans_x = _mm256_sub_ps(undistort_x, _mm256_set1_ps((float)center_frame.x));
        __m256 trans_y = _mm256_sub_ps(undistort_y, _mm256_set1_ps((float)center_frame.y));

        float* arr_trans_x = (float*)&trans_x;
        float* arr_trans_y = (float*)&trans_y;
        for(int j = 0; j < 8; j++){
            float compensated_x = center_frame.x + arr_trans_x[j]*c_roll_comp + arr_trans_y[j]*s_roll_comp;
            float compensated_y = center_frame.y - arr_trans_x[j]*s_roll_comp + arr_trans_y[j]*c_roll_comp;

            float offset_pan = panAngleDeviation(compensated_x);
            float offset_tilt = tiltAngleDeviation(compensated_y);

            float distance_y = verticalDistance(offset_tilt);
            float distance_x = horizontalDistance(distance_y, offset_pan);

            if(sqrt(distance_x*distance_x + distance_y*distance_y) < 55.0 && !ball)//ignore feature in less than 55 cm
                continue;

            float rotated_x = distance_x*c_pan_servo + distance_y*s_pan_servo;
            float rotated_y = -distance_x*s_pan_servo + distance_y*c_pan_servo;

            // Robot local coordinate in real world

            cv::Point local_coord(rotated_x, rotated_y);

            int mapped_x = remap_x + local_coord.x;
            int mapped_y = remap_y + local_coord.y;
#ifdef DEBUG
            curr_idx = i+j;
            sz_bound = (int)(curr_idx < _points.size());
            if(sz_bound)target_points_viz.at<uchar>(_points[curr_idx].y, _points[curr_idx].x) = 255;
#endif
            if(mapped_x < 0 || mapped_x >= POINTS_MAP_W ||
               mapped_y < 0 || mapped_y >= POINTS_MAP_H ||
               points_map.at<uchar>(mapped_y, mapped_x) > 0)
                continue;

            projected_point.push_back(local_coord);
            points_map.at<uchar>(mapped_y, mapped_x) = 255;
        }        
//        (void)arr_x;
//        (void)arr_y;
    }
#ifdef DEBUG
    if(!ball){
        switch(debug_viz_mode){
        case 0:debug_viz_ = target_points_viz.clone();break;
        case 1:debug_viz_ = points_map.clone();break;
        default:break;
        }
    }
#endif

    return projected_point;
}

float Localization::calcShootDir(const cv::Point2f &_ball_pos){//global ball pos
    // cv::Point2f goal_post1((attack_dir_?landmark_pos_[1][0]:landmark_pos_[1][4]) * 100.0f);
    // cv::Point2f goal_post2((attack_dir_?landmark_pos_[1][1]:landmark_pos_[1][5]) * 100.0f);
    constexpr float goal_postx = FIELD_LENGTH + BORDER_STRIP_WIDTH;
    constexpr float goal_posty1 = BORDER_STRIP_WIDTH + (FIELD_WIDTH - GOAL_WIDTH) * .5f;
    constexpr float goal_posty2 = BORDER_STRIP_WIDTH + (FIELD_WIDTH + GOAL_WIDTH) * .5f;
    cv::Point2f goal_post1(goal_postx, goal_posty1);
    cv::Point2f goal_post2(goal_postx, goal_posty2);
    constexpr float END_OF_XMONITOR = (float)(FIELD_LENGTH + (BORDER_STRIP_WIDTH << 1));
    constexpr float center_goal_y = (goal_posty1 + goal_posty2) * .5f;
    cv::Point2f center_goal(END_OF_XMONITOR, center_goal_y);

    constexpr float GK_OCCUPANCY = 80.0f;
    constexpr float HALF_GK_OCCUPANCY = GK_OCCUPANCY * .5f;
    cv::Point2f zero_dir_area1_tl(goal_post1.x - PENALTY_MARK_DISTANCE, goal_post1.y);
    cv::Point2f zero_dir_area1_br(goal_post1.x, center_goal.y - HALF_GK_OCCUPANCY);

    cv::Point2f zero_dir_area2_tl(goal_post2.x - PENALTY_MARK_DISTANCE, center_goal.y + HALF_GK_OCCUPANCY);
    cv::Point2f zero_dir_area2_br(goal_post2.x, goal_post2.y);

//    std::cout << zero_dir_area1_tl << " ; " << zero_dir_area1_br << std::endl;
//    std::cout << zero_dir_area2_tl << " ; " << zero_dir_area2_br << std::endl;

    if((_ball_pos.x > zero_dir_area1_tl.x && _ball_pos.x < zero_dir_area1_br.x &&
        _ball_pos.y > zero_dir_area1_tl.y && _ball_pos.y < zero_dir_area1_br.y) ||
            (_ball_pos.x > zero_dir_area2_tl.x && _ball_pos.x < zero_dir_area2_br.x &&
             _ball_pos.y > zero_dir_area2_tl.y && _ball_pos.y < zero_dir_area2_br.y) ||
            resetting_particle_){
        return 360.0f;
    }

    cv::Point2f gk_avoidance_area_tl(center_goal.x - PENALTY_MARK_DISTANCE - BORDER_STRIP_WIDTH, center_goal.y - HALF_GK_OCCUPANCY);
    cv::Point2f gk_avoidance_area_br(center_goal.x - BORDER_STRIP_WIDTH, center_goal.y + HALF_GK_OCCUPANCY);

//    std::cout << gk_avoidance_area_tl << " ; " << gk_avoidance_area_br << std::endl;

    if((_ball_pos.x >= gk_avoidance_area_tl.x && _ball_pos.x <= gk_avoidance_area_br.x &&
        _ball_pos.y >= gk_avoidance_area_tl.y && _ball_pos.y <= gk_avoidance_area_br.y)){

        float robot_theta = robot_state_.theta;

        if(robot_theta < .0f)robot_theta = 360.0f + robot_theta;
        bool target_cond = robot_theta > 180.0 && robot_theta < 360.0;
        constexpr float HALF_GOAL_WIDTH = GOAL_WIDTH * .5f;
        cv::Point2f target_goal(center_goal.x - BORDER_STRIP_WIDTH, center_goal.y + (target_cond ? -HALF_GOAL_WIDTH : HALF_GOAL_WIDTH));

        float target_dir = atan2(target_goal.y - _ball_pos.y, target_goal.x - _ball_pos.x) * Math::RAD2DEG;
        if(target_dir < .0f)target_dir = 360.0f + target_dir;
        return target_dir;
    }

    //Shooting Dir First Style
//    cv::Point2f avoidance_gk_area1_tl(center_goal.x - PENALTY_MARK_DISTANCE - BORDER_STRIP_WIDTH, center_goal.y - (GK_OCCUPANCY * 0.5f));
//    cv::Point2f avoidance_gk_area1_br(center_goal.x, center_goal.y);

//    cv::Point2f avoidance_gk_area2_tl(center_goal.x - PENALTY_MARK_DISTANCE - BORDER_STRIP_WIDTH, center_goal.y);
//    cv::Point2f avoidance_gk_area2_br(center_goal.x, center_goal.y + (GK_OCCUPANCY * 0.5f));

//    if((_ball_pos.x >= avoidance_gk_area1_tl.x && _ball_pos.x <= avoidance_gk_area1_br.x &&
//        _ball_pos.y >= avoidance_gk_area1_tl.y && _ball_pos.y <= avoidance_gk_area1_br.y)){
//        cv::Point2f target_goal(center_goal.x, center_goal.y + GOAL_WIDTH * 0.5f);
//        float target_dir = atan2(target_goal.y - _ball_pos.y, target_goal.x - _ball_pos.x) * Math::RAD2DEG;
//        if(target_dir < .0f)target_dir = 360.0f + target_dir;
//        return target_dir;
//    }

//    if((_ball_pos.x > avoidance_gk_area2_tl.x && _ball_pos.x < avoidance_gk_area2_br.x &&
//        _ball_pos.y > avoidance_gk_area2_tl.y && _ball_pos.y < avoidance_gk_area2_br.y)){
//        cv::Point2f target_goal(center_goal.x, center_goal.y - GOAL_WIDTH * 0.5f);
//        float target_dir = atan2(target_goal.y - _ball_pos.y, target_goal.x - _ball_pos.x) * Math::RAD2DEG;
//        if(target_dir < .0f)target_dir = 360.0f + target_dir;
//        return target_dir;
//    }

//==============
    float goal_width = fabs(goal_post1.y - goal_post2.y);
    float diff_x1 = _ball_pos.x - goal_post1.x;
    float diff_y1 = _ball_pos.y - goal_post1.y;
    float diff_x2 = _ball_pos.x - goal_post2.x;
    float diff_y2 = _ball_pos.y - goal_post2.y;
    float dist_to_post1 = sqrt(diff_x1*diff_x1 + diff_y1*diff_y1);
    float dist_to_post2 = sqrt(diff_x2*diff_x2 + diff_y2*diff_y2);
    float angle_interval = (dist_to_post1*dist_to_post1 + dist_to_post2*dist_to_post2 - goal_width*goal_width)/
            (2.0f * dist_to_post1*dist_to_post2);
    angle_interval = acos(angle_interval);
//    float center_dir = (attack_dir_?CV_PI - atan2(center_goal.y - _ball_pos.y,_ball_pos.x - center_goal.x):
//      atan2(center_goal.y - _ball_pos.y,center_goal.x - _ball_pos.x))*Math::RAD2DEG;
    float center_dir = atan2(center_goal.y - _ball_pos.y, center_goal.x - _ball_pos.x) * Math::RAD2DEG;
    if(center_dir < .0f)center_dir = 360.0f + center_dir;

//    std::cout << "Center Dir : " << center_dir << std::endl;
    //not yet added random dir, but the interval is already
    return center_dir;
}

void Localization::publishData(){
    vision_utils::Particles particles_msg;
    geometry_msgs::PoseStamped robot_state_msg;
    robot_state_msg.pose.position.x = robot_state_.x ;
    robot_state_msg.pose.position.y = robot_state_.y ;
    robot_state_msg.pose.orientation.z = robot_state_.theta;
    robot_state_msg.header.stamp = this->stamp_;
    robot_state_msg.header.seq++;

    particles_msg.particle = particles_state_;
    particles_msg.header.stamp = this->stamp_;
    particles_msg.header.seq++;

    features_.header.stamp = this->stamp_;
    features_.header.seq++;

    geometry_msgs::PointStamped proj_ball_stamped_msg;
    proj_ball_stamped_msg.header.stamp = this->stamp_;
    proj_ball_stamped_msg.header.seq++;
    geometry_msgs::Point ball_pos_msg;
    ball_pos_msg.z = -1.0;
    proj_ball_stamped_msg.point = ball_pos_msg;

    if(ball_pos_.x != -1.0 &&
        ball_pos_.y != -1.0){
        Points ball_pos;
        ball_pos.emplace_back(cv::Point(ball_pos_.x,ball_pos_.y));
        ball_pos = pointsProjection(ball_pos, true);

        if(ball_pos.size() > 0){

            float c_t = cos(robot_state_.theta * Math::DEG2RAD);
            float s_t = sin(robot_state_.theta * Math::DEG2RAD);
            float shoot_dir = calcShootDir(cv::Point2f(robot_state_.x + ball_pos.front().y*c_t - ball_pos.front().x*s_t,
                                     robot_state_.y + ball_pos.front().y*s_t + ball_pos.front().x*c_t));

            ball_pos_msg.x = ball_pos.front().y;
            ball_pos_msg.y = -ball_pos.front().x;
            ball_pos_msg.z = shoot_dir;
            projected_ball_pub_.publish(ball_pos_msg);

            proj_ball_stamped_msg.point.x = ball_pos.front().x;
            proj_ball_stamped_msg.point.y = ball_pos.front().y;
            proj_ball_stamped_msg.point.z = shoot_dir;
        }
    }

    robot_state_pub_.publish(robot_state_msg);
    particles_state_pub_.publish(particles_msg);
    features_pub_.publish(features_);
    projected_ball_stamped_pub_.publish(proj_ball_stamped_msg);
}

void Localization::process(){
    
    if(!setInputImage())return;    

//    vision_utils::LineTip tip_points;

//    cv::imshow("IG",invert_green_);
//    cv::imshow("SW",segmented_white_);

    constexpr float FRAME_AREA = FRAME_WIDTH*FRAME_HEIGHT;
    bool blind = (1.0f - (float)cv::countNonZero(invert_green_)/FRAME_AREA) < 0.01f; // assume the robot is blind when less than 1% green in frame

    auto t1 = boost::chrono::high_resolution_clock::now();
    lost_features_ = true;
    // std::cout << front_fall_limit_ << " ; " << behind_fall_limit_ << " ; " << right_fall_limit_ << " ; " << left_fall_limit_ << std::endl;
    if(tilt_servo_angle_ < tilt_limit_ &&
            offset_head_ < 40.0f &&
            fabs(roll_offset_) < 22.0f){ //robot condition for feature observation
        Points target_points;
        Vectors3 line_models;

        LocalizationUtils::getInstance()->scanLinePoints(invert_green_, segmented_white_, field_boundary_, target_points);
//        LocalizationUtils::getInstance()->adaptiveScanLinePoints(invert_green_,segmented_white_,tilt_servo_angle_,pan_servo_angle_,field_boundary_,target_points);

#ifdef METHOD_1
#ifdef DEBUG
        Points projected_points = pointsProjection(target_points);
#else
        Points projected_points = fastPointsProjection(target_points);
#endif
        target_points.clear();
        target_points.shrink_to_fit();
        arrangeTargetPoints(projected_points);
        getFeaturesModels(projected_points, line_models, features_);
#else
        arrangeTargetPoints(target_points);
        getFeaturesModels(target_points,line_models,features_,tip_points);
        cv::Mat check_tp = cv::Mat::zeros(FRAME_HEIGHT,FRAME_WIDTH, CV_8UC1);
        for(size_t i = 0;i < target_points.size();i++){
            check_tp.at<uchar>(target_points[i].y, target_points[i].x) = 255;
        }

        for(size_t i=0;i<tip_points.tip1.size();i++){
            cv::line(check_tp, cv::Point(tip_points.tip1[i].x, tip_points.tip1[i].y),
            cv::Point(tip_points.tip2[i].x, tip_points.tip2[i].y),
                cv::Scalar(150),2);
        }
//        cv::imshow("CHECK_TOK",check_tp);
        publishTipPoints(tip_points);
#endif
    }

    update();

    if(blind){
        robot_state_.x = 999.0;
        robot_state_.y = 999.0;
    }

    if(odometer_buffer_.size() > 0)
        odometer_buffer_.erase(odometer_buffer_.begin());

    publishData();

    auto t2 = boost::chrono::high_resolution_clock::now();
    auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();

    static auto sum_et = .0;

#ifdef DEBUG
    std::cout << "Elapsed Time : " << elapsed_time << " ms : SUM ET : " << sum_et << std::endl;
#endif

    if(lost_features_)
        sum_et += elapsed_time;

    if(sum_et > 200.0 || (!lost_features_ && sum_et > .0)){//hold for 200 ms
        features_present_ = 0;
        features_.feature.clear();
        sum_et = .0;
    }
    
    actionForMonitor();
    
#ifdef DEBUG
    if(!debug_viz_.empty())
        cv::imshow("DEBUG_VIZ", debug_viz_);
    cv::waitKey(1);
#endif
}
