import requests
import json
import pandas as pd
from datetime import datetime
import os


class SlackManager:

    def __init__(self):
        self.slack_token = 'xoxb-3812497290450-3812609303618-BAV9NnRzL2U8Fg14mal8lO8a'
        self.slack_channel = '#tag2g'
        self.error_channel = '#tag2g'
        self.slack_icon_emoji = ':man_climbing:'
        self.slack_user_name = 'Training Bot'

    def post_message_to_slack(self, text, blocks = None):
        return requests.post('https://slack.com/api/chat.postMessage', {
            'token': self.slack_token,
            'channel': self.slack_channel,
            'text': text,
            'icon_emoji': self.slack_icon_emoji,
            'username': self.slack_user_name,
            'blocks': json.dumps(blocks) if blocks else None
        }).json()

    def post_error_to_slack(self, text, blocks = None):
        return requests.post('https://slack.com/api/chat.postMessage', {
            'token': self.slack_token,
            'channel': self.error_channel,
            'text': text,
            'icon_emoji': self.slack_icon_emoji,
            'username': self.slack_user_name,
            'blocks': json.dumps(blocks) if blocks else None
        }).json()

    # fnct to be edited and called to post on SLACK about training
    def make_msg(self, header_str, epoch, vq_loss, recon_loss):
        slack_block = [
            {
                "type": "divider"
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "I have completed an epoch!\nEpoch: %d" % (epoch)
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "VQ Loss: %.4f\nRecon loss: %.4f" %(vq_loss, recon_loss)
                }
            }
        ]
        # COMMENT WHEN RUNNING TESTS  >>>
        # info = list(zip([global_count], [reward], [datetime.now()]))
        # df = pd.DataFrame(info, index=[global_count], columns=['ep', 're', 'ts'])
        # if os.path.exists('rewards_sang.csv'):
        #     df.to_csv('rewards_sang.csv', header=False, mode='a')
        # else:
        #     df.to_csv('rewards_sang.csv', header=True, mode='w')
        # <<< COMMENT WHEN RUNNING TESTS
        print('>> Publishing updates on SLACK')
        self.post_message_to_slack(
            header_str,
            slack_block)
        # print(self.post_message_to_slack(
        #     "New action!",
        #

    def post_error(self, header_str):
        slack_block = [
            {
                "type": "divider"
            },
            {
                "type": "divider"
            },
            # {
            #     "type": "section",
            #     "text": {
            #         "type": "mrkdwn",
            #         "text": ":bangbang:\t:bangbang:\t:bangbang: \t\t :bangbang:\t:bangbang:\t:bangbang:"
            #     }
            # },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":thermometer: *The websocket has crashed!* :thermometer:"
                }
            }
        ]

        print('Sending')
        self.post_error_to_slack(
            header_str,
            slack_block)
