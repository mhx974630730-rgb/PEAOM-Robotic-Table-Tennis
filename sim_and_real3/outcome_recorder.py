#!/usr/bin/env python3
"""Keyboard helper for recording mutually exclusive physical-trial outcomes."""

import rospy
from std_msgs.msg import Int32, String


KEYS = {
    's': 'successful_return',
    'm': 'missed_ball',
    'e': 'edge_contact',
    'n': 'net_failure',
    'o': 'out_of_table',
    'r': 'robot_reach',
    'd': 'detection_failure',
    'u': 'unknown',
}


class OutcomeRecorder:
    def __init__(self):
        self.episode = None
        self.publisher = rospy.Publisher(
            '/experiment/trial_outcome', String, queue_size=10
        )
        rospy.Subscriber(
            '/experiment/action_episode', Int32, self._episode_callback, queue_size=1
        )

    def _episode_callback(self, msg):
        self.episode = int(msg.data)

    def run(self):
        print('Physical-trial outcome recorder')
        print('s=successful_return  m=missed_ball  e=edge_contact')
        print('n=net_failure       o=out_of_table r=robot_reach')
        print('d=detection_failure u=unknown      q=quit')
        while not rospy.is_shutdown():
            episode_text = '?' if self.episode is None else str(self.episode)
            raw = input(f'episode {episode_text} > ').strip()
            if not raw:
                continue
            key = raw[0].lower()
            if key == 'q':
                return
            if key not in KEYS:
                print('Unknown key. Use s/m/e/n/o/r/d/u/q.')
                continue
            note = raw[1:].strip()
            payload = KEYS[key] + ('|' + note if note else '')
            self.publisher.publish(String(data=payload))
            print(f'recorded: episode={episode_text}, outcome={KEYS[key]}, note={note}')


if __name__ == '__main__':
    rospy.init_node('physical_trial_outcome_recorder', anonymous=True)
    OutcomeRecorder().run()
