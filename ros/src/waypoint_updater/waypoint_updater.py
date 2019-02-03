#!/usr/bin/env python

import numpy as np
import math
import rospy

from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32


'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # 200 # Number of waypoints we will publish. You can change this number
# maximum brakes applied
MAX_DCL_APPL = .5
# rate in Hz
PUBLISH_RATE = 25

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)


        # additional member variables for WaypointUpdater
        self.pose = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        self.no_of_waypoints = -1

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # not used
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        self.loop()

    def loop(self):
        # periodically pulish final_waypoints
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        # current position of the car
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # index of the closest waypoint
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # check if the closest point is ahead or behind the vehicle
        closest_vector = np.array(self.waypoints_2d[closest_idx])
        previous_vector = np.array(self.waypoints_2d[closest_idx - 1])
        pose_vector = np.array([x,y])

        result = np.dot((closest_vector - previous_vector), (pose_vector - closest_vector))

        if result > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    # publish final set of waypoints as a Lane
    def publish_waypoints(self):
        if self.no_of_waypoints == -1:
            return
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):

        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS

        ## # NOTE: revisit this again..
        if farthest_idx < self.no_of_waypoints:
            base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
            # rospy.logfatal('\nNormal closest_idx: %d, farthest_idx:%d, length of self.base_lane.waypoints :%d', closest_idx, farthest_idx, len(base_waypoints))
        else:
            rospy.logfatal("WP lookahead passed end of track!!!")
            offset = farthest_idx - self.no_of_waypoints
            farthest_idx = self.no_of_waypoints - 2
            base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

            base_waypoints = base_waypoints + self.base_lane.waypoints[0:offset]
            # rospy.logfatal('\nError! closest_idx: %d, farthest_idx:%d, length of self.base_lane.waypoints :%d', closest_idx, farthest_idx, len(base_waypoints))


        #if(len(base_waypoints) != LOOKAHEAD_WPS):
            #rospy.logfatal('\nError! closest_idx: %d, farthest_idx:%d, length of self.base_lane.waypoints :%d', closest_idx, farthest_idx, len(base_waypoints))

        rospy.loginfo('\nwpupd-gen-lane: generate_lane called - closest_idx:%d, farthest_idx:%d, stopline_wp_idx:%d', closest_idx, farthest_idx, self.stopline_wp_idx)
        # keep the base waypoints as final_waypoints if the traffic light is
        # not in sight or too far
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
            # rospy.logfatal('\nwp-genlane: Stop light detected,but out of range')
            lane.waypoints = base_waypoints
        else:
            # or else, decelerate_waypoint velocities accordingly
            rospy.loginfo('\nwp-genlane: decelerating!!')
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        #rospy.logfatal('\nwpupd-decel: Enter decelerate_waypoints with waypoints length:%d closet idx : %d', len(waypoints), closest_idx)
        #rospy.logfatal('stopline idx is %d and closest idx is %d',self.stopline_wp_idx, closest_idx)
        temp = []
        count = 0
        for i, wp in enumerate(waypoints):
            # get the currently available base waypoint positions
            p = Waypoint()
            p.pose = wp.pose

            #rospy.logfatal('waypoint x y z is %d', wp.pose.pose.position.x)
            # calculate stop waypoint after accounting for the cars front length
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)

            # rospy.loginfo('\nwp-decel: stop index :%d ', stop_idx)
            # rospy.loginfo('\nwp-decel: distance => wp1:%d and wp2:%d ', i, stop_idx)

            #stop line distance
            dist = self.distance(waypoints, i, stop_idx)

            if dist <= 1:
                ## if the vehicle is too close, try to stop immediately
                vel = 0
            elif dist <=5:
                ## apply more brake is the distance is shorter
                vel = math.sqrt(2 * MAX_DCL_APPL * dist)
            else:
                ## apply brake based on the distance availalbe
                vel = wp.twist.twist.linear.x - (wp.twist.twist.linear.x/dist)

            if vel < 1.:
                vel = 0.

            # if the calculated velocity goes above what the car is driving at,
            # no need to update the velocity
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp


    def pose_cb(self, msg):
        # current pose of the vehicle received from the simulator/vehicle
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.no_of_waypoints = len(waypoints.waypoints)
        self.base_lane = waypoints
        if not self.waypoints_2d:
            # construct 2d co-ordinates
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)


    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
