# utils/wechat_notify.py

import requests
import json

def send_wechat_template(to_users, title, content1, content2, content3, remark):
    """
    发送微信模板消息

    :param to_users: List[str] 接收用户 openid 列表
    :param title: str 标题内容
    :param content1: str 内容字段1
    :param content2: str 内容字段2
    :param content3: str 内容字段3
    :param remark: str 备注内容
    """
    for to_user in to_users:
        template_data = {
            "to_user": to_user,
            "template_id": "nyQ-0vYb0bl5EZWT2OK8jX46NNsnrzWXxminYjO2Y8A",
            "data": {
                "thing4": title,
                "thing31": content1,
                "thing40": content2,
                "thing5": content3,    # ✅ 这里改成传进来的 content3
                "remark": remark
            },
            "url": "https://cp.zuai.me",
            "url_params": {
                "order_id": "395248",
                "user": "苏"
            }
        }

        # 打印一下请求数据，方便调试
        print("🚀 准备发送微信提醒，发送内容如下：")
        print(json.dumps(template_data, ensure_ascii=False, indent=2))

        try:
            url = "http://134.175.237.107:5001/send_template"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": "sw63828"
            }
            response = requests.post(url, headers=headers, data=json.dumps(template_data))
            response.raise_for_status()
            response_json = response.json()
            print(f"✅ 微信提醒已发送给用户 {to_user}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 微信提醒发送失败: {e}")
        except ValueError:
            print(f"❌ 微信提醒返回格式错误")


import pandas as pd

if __name__ == "__main__":
    # === 自动读最新预测结果 ===
    df = pd.read_csv("data/next_predict_result.csv", sep="|")
    last_row = df.iloc[-1]

    issue = last_row["issue"]
    pred_dan = last_row["pred_dandan"]
    pred_digits = last_row["pred_digits"]
    hit = last_row["hit"]

    title = f"📢 福彩3D拼搏{issue}期v3"
    content1 = f"独胆预测：{pred_dan}"
    content2 = f"试机号预测：{pred_digits}"
    content3 = f"预测状态：{hit}"
    remark = "👉 请尽快查看，祝好运！"

    # 这里替换成你要通知的 openid 列表
    to_users = ["oXUv66MibUi7VInLBf7AHqMIY438"]

    send_wechat_template(to_users, title, content1, content2, content3, remark)
