#include "v9_localization/v9_amcl.h"

AMCL::AMCL():
    rand_gen_(std::random_device{}()),
    features_present_(0),
    short_term_avg_(.0f),
    long_term_avg_(.0f),
    last_weight_avg_(.0f),
    resetting_particle_(false){

    robot_state_.x = robot_state_.y = (BORDER_STRIP_WIDTH>>1);
    robot_state_.theta = .0;
    last_robot_state_ = robot_state_;
    odometer_out_.x = odometer_out_.y = .0;

}

void AMCL::initializeParticles(){
    particles_state_.resize(params_.num_particles);
    float uniform_weight = 1.0f/params_.num_particles;
    for(Particles::iterator it=particles_state_.begin();
        it!=particles_state_.end();it++){
        it->x = genUniIntDist(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_LENGTH);
        it->y = genUniIntDist(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_WIDTH);
        it->z = genUniIntDist(0, 360);
        it->w = uniform_weight;
    }
}

Particles AMCL::resamplingWheel(const Particles &_old_particles, Particle &_best_particle){
    Particles new_particles(params_.num_particles);
    Particle best_particle;
    int index = genUniIntDist(0, params_.num_particles-1);
    float beta = .0f;
    float max_weight = .0f;
    float prob = std::max(.0f, 1.0f - short_term_avg_/long_term_avg_);
    resetting_particle_ = prob > .25f;
    for(Particles::const_iterator it=_old_particles.begin();
        it != _old_particles.end(); it++){
        if(it->w > max_weight){
            max_weight = it->w;
            best_particle.x = it->x;
            best_particle.y = it->y;
            best_particle.z = it->z;
            best_particle.w = it->w;
        }
    }

    _best_particle = best_particle;

    cv::Vec2i x_interval;
    cv::Vec2i y_interval;
    switch(features_present_){
        case 1:
            x_interval[0] = last_robot_state_.x - 100;
            x_interval[1] = last_robot_state_.x + 100;
            y_interval[0] = last_robot_state_.y - 100;
            y_interval[1] = last_robot_state_.y + 100;
            prob = std::min(.05f, prob);
            break;
        default:
            x_interval[0] = BORDER_STRIP_WIDTH;
            x_interval[1] = BORDER_STRIP_WIDTH + FIELD_LENGTH;
            y_interval[0] = BORDER_STRIP_WIDTH;
            y_interval[1] = BORDER_STRIP_WIDTH + FIELD_WIDTH;
            break;
    }

    for(int i = 0; i < params_.num_particles; i++){
        if(genUniRealDist(.0, 1.0) < prob){
            new_particles[i].x = genUniIntDist(x_interval[0], x_interval[1]);
            new_particles[i].y = genUniIntDist(y_interval[0], y_interval[1]);
            new_particles[i].z = genUniIntDist(0, 360);
            new_particles[i].w = 0;
        }else{
            beta += genUniRealDist(.0, 2.0 * max_weight);

            while(beta > _old_particles[index].w){
                beta -= _old_particles[index].w;
                index = (index+1)%params_.num_particles;
            }
            new_particles[i] = _old_particles[index];
        }

    }

    return new_particles;
}

Particles AMCL::lowVarianceResampling(const Particles &_old_particles, Particle &_best_particle){
    Particles new_particles(params_.num_particles);
    Particle best_particle;
    float max_weight = .0f;
    for(Particles::const_iterator it=_old_particles.begin();
        it != _old_particles.end(); it++){
        if(it->w > max_weight){
            max_weight = it->w;
            best_particle.x = it->x;
            best_particle.y = it->y;
            best_particle.z = it->z;
            best_particle.w = it->w;
        }
    }

    _best_particle = best_particle;

    float r = genUniRealDist(0.0, 1.0/params_.num_particles);
    float c = _old_particles.front().w;
    int idx = 0;

    float prob = std::max(.0f, 1.0f - short_term_avg_/long_term_avg_);
    resetting_particle_ = prob > .25f;
    cv::Vec2i x_interval;
    cv::Vec2i y_interval;

    switch(features_present_){
        case 1:
            x_interval[0] = last_robot_state_.x - 100;
            x_interval[1] = last_robot_state_.x + 100;
            y_interval[0] = last_robot_state_.y - 100;
            y_interval[1] = last_robot_state_.y + 100;
            prob = std::min(.05f, prob);
            break;
        default:
            x_interval[0] = BORDER_STRIP_WIDTH;
            x_interval[1] = BORDER_STRIP_WIDTH + FIELD_LENGTH;
            y_interval[0] = BORDER_STRIP_WIDTH;
            y_interval[1] = BORDER_STRIP_WIDTH + FIELD_WIDTH;
            break;
    }

    for(int i=0;i<params_.num_particles;i++){
        if(genUniRealDist(.0, 1.0) < prob){
            new_particles[i].x = genUniIntDist(x_interval[0], x_interval[1]);
            new_particles[i].y = genUniIntDist(y_interval[0], y_interval[1]);
            new_particles[i].z = genUniIntDist(0, 360);
            new_particles[i].w = .0;
        }else{
            float U = r + (float)i/params_.num_particles;
            while(U > c){
                idx = (idx+1)%params_.num_particles;
                c += _old_particles[idx].w;
            }
            new_particles[i] = _old_particles[idx];
        }
    }

    return new_particles;
}

void AMCL::calcRobotPose(const Particle &_best_particle){
    float total_sin = .0f;
    float total_cos = .0f;
    float total_x = .0f;
    float total_y = .0f;
//    for(Particles::iterator it=particles_state_.begin();
//        it != particles_state_.end(); it++){
//        total_x += it->x;
//        total_y += it->y;
//        float orientation_rad = it->z * Math::DEG2RAD;
//        total_sin += sin(orientation_rad);
//        total_cos += cos(orientation_rad);
//    }


    int centered_particles = 0;
//    float particle_std_dev = .0f;
//    std::cout << "BP : " << _best_particle.x << " ; " << _best_particle.y << std::endl;
//    float particle_std_dev = 0;
//    for(Particles::iterator it=particles_state_.begin();
//        it!=particles_state_.end();it++){
//        float diff_x = _best_particle.x - it->x;
//        float diff_y = _best_particle.y - it->y;
//        float dist2best_particle = sqrt(diff_x*diff_x + diff_y*diff_y);
////        particle_std_dev += dist2best_particle;
//        if(dist2best_particle < 20){
//            centered_particles++;
//            total_x += it->x;
//            total_y += it->y;
//            float orientation_rad = it->z * Math::DEG2RAD;
//            total_sin += sin(orientation_rad);
//            total_cos += cos(orientation_rad);
//        }
//    }

    int curr_idx = 0;
    int sz_bound = 0;

    for(int i = 0;i < particles_state_.size();i+=8){
        auto mem1 = boost::alignment::aligned_alloc(32, 64 * sizeof(float));
        float* mem1_specific = new(mem1) float;
        auto mem2 = boost::alignment::aligned_alloc(32, 64 * sizeof(float));
        float* mem2_specific = new(mem2) float;
        std::unique_ptr<float[], boost::alignment::aligned_delete > arr_pstate_x(mem1_specific);
        std::unique_ptr<float[], boost::alignment::aligned_delete > arr_pstate_y(mem2_specific);

        for(int j=0;j < 8;j++){
            curr_idx = i + j;
            sz_bound = (int)(curr_idx < particles_state_.size());
            curr_idx *= sz_bound;
            arr_pstate_x.get()[j] = sz_bound * particles_state_[curr_idx].x;
            arr_pstate_y.get()[j] = sz_bound * particles_state_[curr_idx].y;
        }

        __m256 pstate_x = _mm256_load_ps(arr_pstate_x.get());
        __m256 pstate_y = _mm256_load_ps(arr_pstate_y.get());
        __m256 diff_x = _mm256_sub_ps(_mm256_set1_ps(_best_particle.x), pstate_x);
        __m256 diff_y = _mm256_sub_ps(_mm256_set1_ps(_best_particle.y), pstate_y);
        __m256 dist2best_particle = _mm256_sqrt_ps(
                                    _mm256_add_ps(
                                    _mm256_mul_ps(diff_x, diff_x),
                                    _mm256_mul_ps(diff_y, diff_y)));
        float* d2bp = (float*)&dist2best_particle;
        for(int j=0; j < 8;j++){
            curr_idx = i+j;
//            particle_std_dev += d2bp[j];
            if(d2bp[j] < 25.0f && curr_idx < particles_state_.size()){
                ++centered_particles;
                total_x += particles_state_[curr_idx].x;
                total_y += particles_state_[curr_idx].y;
                float orientation_rad = particles_state_[curr_idx].z * Math::DEG2RAD;
                total_sin += sin(orientation_rad);
                total_cos += cos(orientation_rad);
            }
        }
    }

//    if(centered_particles > .35 * params_.num_particles){// if there are 35% particle distributed around
//        robot_state_.x = total_x/centered_particles;
//        robot_state_.y = total_y/centered_particles;
//        robot_state_.theta = atan2(total_sin,total_cos) * Math::RAD2DEG;
//    }else{
//        robot_state_ = last_robot_state_;
//    }

//    particle_std_dev /= params_.num_particles;

//    std::cout << "Particle STD DEV : " << particle_std_dev << std::endl;
//    std::cout << "Centered Particles : " << centered_particles << std::endl;

    if(resetting_particle_){
        robot_state_.x = 999.0;
        robot_state_.y = 999.0;
    }else if(centered_particles > (.3f * (float)params_.num_particles)){
        robot_state_.theta = atan2(total_sin,total_cos) * Math::RAD2DEG;
        robot_state_.x = total_x/centered_particles;
        robot_state_.y = total_y/centered_particles;
        last_robot_state_ = robot_state_;
    }else{
        float diff_x = .0f;
        float diff_y = .0f;
        if(odometer_buffer_.size() > 0){
            diff_x = odometer_buffer_.front().x;
            diff_y = -odometer_buffer_.front().y;
        }
        float trans = sqrt(diff_x*diff_x + diff_y*diff_y);
        robot_state_.theta = atan2(total_sin, total_cos);
        robot_state_.x = last_robot_state_.x + trans * cos(robot_state_.theta);
        robot_state_.y = last_robot_state_.y + trans * sin(robot_state_.theta);
        robot_state_.theta *= Math::RAD2DEG;
        last_robot_state_ = robot_state_;     
    }

    resetting_particle_ = false;

//    std::cout << "CENTERED : " << centered_particles << std::endl;
//    if(centered_particles < ((float)params_.num_particles * 0.15)){
//        robot_state_.x = 999;
//        robot_state_.y = 999;
//    }
    
}

void AMCL::update(){
    static Particle best_particle;
    static bool first = true;
    
    sampleMotionModelOdometry(particles_state_, odometer_out_);

    if(features_present_ > 0){
        float weight_avg = .0f;
        measurementModel(particles_state_, features_, weight_avg);
        if(std::isnan(weight_avg)){
            weight_avg = last_weight_avg_;
        }else{
            if(first){
                short_term_avg_ = weight_avg;
                long_term_avg_ = weight_avg;
                first = false;
            }
            last_weight_avg_ = weight_avg;
        }
 #ifdef DEBUG
        std::cout << Color::RED << "Weight Avg. : " << weight_avg << Color::RESET << std::endl;
        std::cout << Color::GREEN << "Short Term Avg. : " << short_term_avg_ << std::endl;
        std::cout << "Long Term Avg. : " << long_term_avg_ << Color::RESET << std::endl;
 #endif
        short_term_avg_ += params_.short_term_rate*(weight_avg - short_term_avg_);
        long_term_avg_ += params_.long_term_rate*(weight_avg - long_term_avg_);

        particles_state_ = resamplingWheel(particles_state_, best_particle);
        //particles_state_ = lowVarianceResampling(particles_state_, best_particle);
    }

    calcRobotPose(best_particle);
}
