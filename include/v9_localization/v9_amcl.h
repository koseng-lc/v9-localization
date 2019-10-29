/*
 *  Author : koseng (Lintang)
*/

#pragma once

#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose2D.h>

#include <v9_localization/LocalizationParamsConfig.h>

#include <vision_utils/vision_common.h>
#include <vision_utils/Particles.h>
#include <vision_utils/Features.h>

#include <opencv2/core/core.hpp>

#include <random>

#include <memory>
#include <boost/align/aligned_alloc.hpp>
#include <boost/align/aligned_delete.hpp>

//#define DEBUG
 #define GAZEBO

using Field::BORDER_STRIP_WIDTH;
using Field::FIELD_LENGTH;
using Field::FIELD_WIDTH;
using Field::GOAL_AREA_LENGTH;
using Field::GOAL_AREA_WIDTH;
using Field::CENTER_CIRCLE_DIAMETER;

typedef geometry_msgs::Quaternion Particle;
typedef std::vector<Particle> Particles;

class AMCL{
public:
    AMCL();

    void initializeParticles();
    void update();

    Particles particles_state_;
    geometry_msgs::Pose2D robot_state_;
    geometry_msgs::Pose2D last_robot_state_;
    geometry_msgs::Pose2D odometer_out_;
    std::vector<geometry_msgs::Pose2D > odometer_buffer_;

    vision_utils::Features features_;

    inline float probDensityFunc(float _x_min_mean, float _variance){
        // look at the reciprocal term
        return (1.0f/sqrt(Math::TWO_PI*_variance))*exp(-.5f*(_x_min_mean*_x_min_mean)/_variance);
    }

    inline float expWeight(float _x_min_mean, float _variance){
        return exp(-.5f * (_x_min_mean * _x_min_mean) / _variance);
    }

    inline float pdfRange(float _x_min_mean){
        // return probDensityFunc(_x_min_mean, params_.range_var);
        return expWeight(_x_min_mean, params_.range_var);
    }

    inline float pdfBeam(float _x_min_mean){
        // return probDensityFunc(_x_min_mean, params_.beam_var);
        return expWeight(_x_min_mean, params_.beam_var);
    }

    inline int genUniIntDist(int _bottom_limit, int _upper_limit){
        std::uniform_int_distribution<int> distr(_bottom_limit, _upper_limit);
        return distr(rand_gen_);
    }

    inline double genUniRealDist(double _bottom_limit, double _upper_limit){
        std::uniform_real_distribution<double> distr(_bottom_limit, _upper_limit);
        return distr(rand_gen_);
    }

    inline double genNormalDist(double _mean, double _variance){
        std::normal_distribution<double> distr(_mean, _variance);
        return distr(rand_gen_);
    }

    // Approximation
    inline float sampleNormal(float _variance){
        float result = .0f;
        float std_dev = sqrt(_variance);
        for(int i=12;i--;){
            result += genUniRealDist(-std_dev,std_dev);
        }
        return .5f*result;
    }

protected:
    virtual void sampleMotionModelOdometry(Particles &_particles_state,
                                           const geometry_msgs::Pose2D &_odometer_out) = 0;
    virtual void measurementModel(Particles &_particles_state,
                                  vision_utils::Features _features_arg,
                                  float &_weight_avg) = 0;
    int features_present_;
    float last_weight_avg_;
    bool resetting_particle_;
    v9_localization::LocalizationParamsConfig params_;
private:

    Particles resamplingWheel(const Particles &_old_particles, Particle &_best_particle);
    Particles lowVarianceResampling(const Particles &_old_particles, Particle &_best_particle);
    void calcRobotPose(const Particle &_best_particle);

    std::mt19937 rand_gen_;

    float short_term_avg_;
    float long_term_avg_;



};
