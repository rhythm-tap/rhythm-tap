



# 環境
from myapp.env import *
# from myapp.env_local import *




# import
import datetime




# サイト情報
SITE_NAME = "Rhythm Tap"
COPY_RIGHT = "Copyright©︎ {year} Yuto All Right Reserved.".format(year=datetime.datetime.now().strftime('%Y'))


# SEO対策
META_TITLE = SITE_NAME
META_DESCRIPTION = ""
META_KEYWORDS = "ウェブサイト, サンプル"
META_TYPE = "website"
META_URL = HTTP_USER
META_LOCALE = "ja_JP"
META_SITE_NAME = SITE_NAME
META_TWITTER_ID = ""
META_TWITTER_CARD = "summary_large_image"
META_TWITTER_IMAGE = HTTP_USER+"static/src/twitter_card.png"






# 現在時刻(キャッシュ対策用)
NOW = datetime.datetime.now().strftime('%Y%m%d%H%M%S');
