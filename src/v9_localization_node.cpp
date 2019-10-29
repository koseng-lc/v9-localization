#include "v9_localization/v9_localization.h"

int main(int argc, char **argv){
    ros::init(argc, argv, "v9_localization_node");

    Localization localization;

    ros::Rate loop_rate(30);
    while(ros::ok()){

        localization.process();

        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}
