"""
多模态信息提取模块
支持图片/视频输入，通过 Ollama qwen3.5 多模态模型提取关键信息
视频处理：faster-whisper 语音转录（主线）+ 场景切换抽帧（辅线）
"""
import os
import sys
import base64
import tempfile
import subprocess
import time
import json
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests

# Selenium 延迟导入（避免启动时必须安装）
_uc_available = None

def _check_uc():
    global _uc_available
    if _uc_available is None:
        try:
            import undetected_chromedriver as uc  # noqa: F401
            _uc_available = True
        except ImportError:
            _uc_available = False
    return _uc_available

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_ENABLED, WHISPER_MODEL


# ============================================================
# Prompt 模板
# ============================================================

IMAGE_EXTRACTION_PROMPT = """请仔细观察这张图片，尽可能详尽地提取其中所有与游戏相关的信息。

请逐条列出以下内容（如果有的话）：
- 游戏中涉及的游戏名称、版本
- 具体活动/事件名称和类型（联动/更新/福利/赛事等）
- 活动时间、地点、参与方式、规则
- 涉及的英雄、皮肤、角色、道具、装备名称
- 奖励内容、获取条件、兑换方式
- 画面中的文字信息（公告、弹幕、评论区、UI数值）
- 玩家反馈和评价倾向
- 任何其他可能有用的事件细节

请用中文逐条列出，尽可能详尽，不要遗漏任何信息。"""

VIDEO_EXTRACTION_PROMPT = """以下是视频的语音转录文本：
{transcript}

以下是视频的关键画面帧。请结合语音和画面，尽可能详尽地提取其中所有与游戏相关的信息。

请逐条列出以下内容（如果有的话）：
- 游戏中涉及的游戏名称、版本
- 主播/视频中讨论的核心话题和观点
- 具体活动/事件名称和类型（联动/更新/平衡调整/赛事等）
- 活动时间、地点、参与方式、规则细节
- 涉及的英雄、皮肤、角色、装备、道具名称
- 奖励内容、获取条件、兑换限制
- 主播/玩家对内容的评价、态度、争议点
- 画面中的重要文字、数据、公告内容
- 玩家评论区的核心观点和争议
- 任何其他可能有用的事件细节

请用中文逐条列出，尽可能详尽，不要遗漏任何信息。"""


class MultimodalExtractor:
    """多模态信息提取器"""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        whisper_model: str = None
    ):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model = model or OLLAMA_MODEL
        self.whisper_model_size = whisper_model or WHISPER_MODEL
        self._whisper = None  # 延迟加载

    # ============================================================
    # 统一入口
    # ============================================================

    def extract(self, media_path: str = None, video_url: str = None, focus: str = "") -> str:
        """
        统一提取入口：支持本地文件路径和视频 URL

        Args:
            media_path: 本地图片/视频文件路径
            video_url: 视频链接
            focus: 产品关注点（如"王者荣耀"），用于过滤无关内容

        Returns:
            提取的文本描述
        """
        self._focus = focus
        # 优先处理 URL
        if video_url and video_url.strip():
            result = self._download_video(video_url.strip())
            if not result:
                return ""
            # 如果返回值看起来不像文件路径，就是已提取好的文本（如笔记）
            if not os.path.exists(result):
                return result
            try:
                print(f"[MultimodalExtractor] 下载完成，开始处理: {result}")
                return self._extract_from_video(result)
            finally:
                self._cleanup_temp(result)

        # 处理本地文件
        if not media_path or not os.path.exists(media_path):
            print(f"[MultimodalExtractor] 文件不存在: {media_path}")
            return ""

        ext = Path(media_path).suffix.lower()
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

        try:
            if ext in image_exts:
                print(f"[MultimodalExtractor] 处理图片: {media_path}")
                return self._extract_from_image(media_path)
            elif ext in video_exts:
                print(f"[MultimodalExtractor] 处理视频: {media_path}")
                return self._extract_from_video(media_path)
            else:
                print(f"[MultimodalExtractor] 不支持的文件类型: {ext}")
                return ""
        except Exception as e:
            print(f"[MultimodalExtractor] 提取失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    # ============================================================
    # 视频链接下载
    # ============================================================

    def _download_video(self, url: str) -> Optional[str]:
        """
        从链接按需下载视频（自动处理各平台 URL 格式差异）
        """
        # 微博链接：先尝试专用提取器，失败则回退 yt-dlp
        if self._is_weibo_url(url):
            result = self._download_weibo(url)
            if result:
                return result
            print("[MultimodalExtractor] 微博专用提取器失败，尝试 yt-dlp...")
            return self._download_with_ytdlp(url)

        # 抖音链接：Selenium 直接抓视频 URL → 下载
        if self._is_douyin_url(url):
            return self._download_douyin_direct(url)

        # 直链
        direct_video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.ts')
        url_lower = url.split('?')[0].lower()
        is_direct = any(url_lower.endswith(ext) for ext in direct_video_exts)

        if is_direct:
            return self._download_direct(url)

        # 其他平台：yt-dlp
        return self._download_with_ytdlp(url)

    # ============================================================
    # 抖音 URL 标准化
    # ============================================================

    def _get_cookies_via_selenium(self, url: str) -> Optional[str]:
        """
        使用 undetected Chrome 获取 Douyin Cookie

        启动反检测的 headless Chrome，访问抖音页面，让浏览器自动完成
        JS 验证挑战并获取完整的 cookie（包括 yt-dlp 需要的 s_v_web_id）。
        将 cookie 保存为 Netscape 格式供 yt-dlp 使用。
        """
        if not _check_uc():
            print("[MultimodalExtractor] undetected-chromedriver 未安装")
            return None

        import undetected_chromedriver as uc
        import tempfile

        # 确定 Chrome 主版本
        try:
            result = subprocess.run(
                [r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--version'],
                capture_output=True, text=True, timeout=5
            )
            version = result.stdout.strip()
            major = int(version.split()[-1].split('.')[0]) if version else 147
        except Exception:
            major = 147

        print(f"[MultimodalExtractor] 启动 undetected Chrome (v{major})...")
        driver = None
        try:
            options = uc.ChromeOptions()
            options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            driver = uc.Chrome(headless=True, use_subprocess=True, version_main=major, options=options)

            # 访问抖音首页获取基础 cookie，然后访问视频页
            driver.get('https://www.douyin.com/')
            time.sleep(3)
            driver.get(url)
            time.sleep(8)  # 等 JS 验证完成和视频加载

            # 提取该域名的所有 cookie
            selenium_cookies = driver.get_cookies()
            domain = urlparse(url).netloc
            cookies = {}
            for c in selenium_cookies:
                name = c.get('name', '')
                value = c.get('value', '')
                if name and value:
                    cookies[name] = value

            driver.quit()
            driver = None

            if not cookies:
                print("[MultimodalExtractor] Selenium 未获取到 Cookie")
                return None

            print(f"[MultimodalExtractor] Selenium 获取到 {len(cookies)} 个 Cookie")
            return self._save_cookies_netscape(cookies, domain)

        except Exception as e:
            print(f"[MultimodalExtractor] Selenium Cookie 获取失败: {e}")
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            return None

    def _download_douyin_direct(self, url: str) -> Optional[str]:
        """
        抖音专用下载：Selenium 打开页面 → 从网络日志抓取视频 mp4 URL → 直接下载

        完全绕过 yt-dlp，因为 yt-dlp 的 DouyinIE 有 cookie 兼容问题。
        """
        import undetected_chromedriver as uc
        import tempfile

        normalized = self._normalize_douyin_url(url)
        if normalized != url:
            print(f"[MultimodalExtractor] 抖音 URL 标准化: {normalized}")

        major = self._get_chrome_major_version()

        print(f"[MultimodalExtractor] 启动 undetected Chrome 提取抖音视频...")
        driver = None
        try:
            options = uc.ChromeOptions()
            options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            driver = uc.Chrome(headless=True, use_subprocess=True, version_main=major, options=options)

            driver.get('https://www.douyin.com/')
            time.sleep(2)
            driver.get(normalized)
            time.sleep(8)
            self._dismiss_overlays(driver)
            time.sleep(2)

            # 从 performance logs 中提取视频 mp4 URL
            logs = driver.get_log('performance')
            video_url = None
            audio_url = None

            for entry in logs:
                try:
                    msg = json.loads(entry['message'])['message']
                    if msg.get('method') == 'Network.responseReceived':
                        resp_url = msg['params']['response']['url']
                        mime = msg['params']['response']['mimeType']
                        if 'video/mp4' in mime and 'douyinvod.com' in resp_url:
                            # 视频轨（有视频内容，通常更大）
                            if 'media-video' in resp_url or 'video' in resp_url.split('/')[-2]:
                                video_url = resp_url
                            # 音频轨（纯音频）
                            elif 'media-audio' in resp_url or 'audio' in resp_url.split('/')[-2]:
                                audio_url = resp_url
                            # 综合流（音视频合一）
                            elif not video_url:
                                video_url = resp_url
                except Exception:
                    continue

            # 保存当前 URL（quit 前获取）
            current_url = driver.current_url
            driver.quit()
            driver = None

            # 图文笔记：提取文本+图片内容
            if '/note/' in current_url:
                print("[MultimodalExtractor] 检测到抖音图文笔记，提取文本内容...")
                return self._extract_note_content(current_url)

            # 优先下载视频（含音轨的），其次单独视频轨
            download_url = video_url or audio_url
            if not download_url:
                # 可能是搜索页/非视频页
                if '/search/' in current_url or '/jingxuan/' in current_url:
                    print("[MultimodalExtractor] 当前页面是搜索/列表页，不是视频页")
                    print("[MultimodalExtractor] 请提供具体视频链接（URL 中包含 /video/{id} 或 modal_id={id}）")
                else:
                    print("[MultimodalExtractor] 未从网络日志中找到视频 URL")
                return None

            print(f"[MultimodalExtractor] 获取到视频 URL: {download_url[:100]}...")

            # 下载到临时文件
            tmp = tempfile.NamedTemporaryFile(suffix='_ytdl', delete=False)
            tmp.close()

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.douyin.com/',
            }
            resp = requests.get(download_url, headers=headers, stream=True, timeout=120)
            resp.raise_for_status()

            total = 0
            max_bytes = 300 * 1024 * 1024
            with open(tmp.name, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        break

            print(f"[MultimodalExtractor] 抖音视频下载完成: {total / 1024 / 1024:.1f} MB（临时文件）")
            return tmp.name

        except ImportError:
            print("[MultimodalExtractor] undetected-chromedriver 未安装")
            return None
        except Exception as e:
            print(f"[MultimodalExtractor] 抖音直接下载失败: {e}")
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
            return None

    @classmethod
    def _resolve_short_link(cls, url: str) -> Optional[str]:
        """展开 v.douyin.com 短链接，获取真实视频 URL"""
        try:
            resp = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }, allow_redirects=True, timeout=15)
            final = resp.url
            if final != url:
                return final
        except Exception as e:
            print(f"[MultimodalExtractor] 短链接展开失败: {e}")
        return url

    @staticmethod
    def _dismiss_overlays(driver):
        """关闭抖音页面上可能遮挡内容的登录弹窗/遮罩层"""
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.by import By

        try:
            # 方法1：按 Escape 键（最通用的关闭弹窗方式）
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
            time.sleep(0.5)
        except Exception:
            pass

        try:
            driver.execute_script('''
                // 方法2：查找并点击关闭按钮
                var closeSelectors = [
                    '[class*="close"]', '[class*="Close"]',
                    '[class*="login-modal-close"]', '[class*="account-close"]',
                    '[data-e2e="close"]', '.dy-account-close',
                    '[aria-label="关闭"]', '[aria-label="close"]',
                    '.login-modal-close', '.modal-close', '.semi-modal-close',
                    '.semi-button', '[class*="cancel"]',
                ];
                for (var i = 0; i < closeSelectors.length; i++) {
                    var els = document.querySelectorAll(closeSelectors[i]);
                    for (var j = 0; j < els.length; j++) {
                        if (els[j].offsetParent !== null) {
                            els[j].click();
                            return 'clicked: ' + closeSelectors[i];
                        }
                    }
                }

                // 方法3：移除所有高 z-index 的遮罩/弹窗
                var allDivs = document.querySelectorAll('div, section, aside');
                var removed = 0;
                for (var i = 0; i < allDivs.length; i++) {
                    try {
                        var style = window.getComputedStyle(allDivs[i]);
                        var zIndex = parseInt(style.zIndex) || 0;
                        var pos = style.position;
                        if ((zIndex > 100 || pos === 'fixed') &&
                            (allDivs[i].className.indexOf('login') >= 0 ||
                             allDivs[i].className.indexOf('Login') >= 0 ||
                             allDivs[i].className.indexOf('mask') >= 0 ||
                             allDivs[i].className.indexOf('overlay') >= 0 ||
                             allDivs[i].className.indexOf('modal') >= 0 ||
                             allDivs[i].className.indexOf('semi') >= 0 ||
                             allDivs[i].id.indexOf('login') >= 0 ||
                             allDivs[i].id.indexOf('modal') >= 0)) {
                            allDivs[i].remove();
                            removed++;
                        }
                    } catch(e) {}
                }

                // 方法4：恢复滚动
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';
                document.body.style.position = '';
                return 'removed: ' + removed;
            ''')
            time.sleep(0.5)
        except Exception:
            pass  # 关闭弹窗失败不阻塞主流程

    def _get_chrome_major_version(self) -> int:
        """获取已安装 Chrome 的主版本号"""
        try:
            result = subprocess.run(
                [r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--version'],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip().split()[-1].split('.')[0]) if result.stdout else 147
        except Exception:
            return 147

    def _extract_note_content(self, note_url: str) -> Optional[str]:
        """提取抖音图文笔记的全部内容（文字+多张图片幻灯片）"""
        import undetected_chromedriver as uc

        major = self._get_chrome_major_version()

        driver = None
        try:
            driver = uc.Chrome(headless=True, use_subprocess=True, version_main=major)
            driver.get(note_url)
            time.sleep(6)

            # 多次尝试关闭登录弹窗（有些弹窗是延迟出现的）
            for attempt in range(3):
                self._dismiss_overlays(driver)
                time.sleep(1.5)

            # 提取页面所有文本
            text = driver.execute_script('''
                var parts = [];

                // 1. 尝试获取笔记描述
                var descSel = '[data-e2e="note-desc"], [data-e2e="detail-desc"], .note-desc, ' +
                              '.detail-desc, .desc-container, [class*="desc"]';
                document.querySelectorAll(descSel).forEach(function(el) {
                    var t = el.innerText ? el.innerText.trim() : '';
                    if (t.length > 10) parts.push(t);
                });

                // 2. 尝试 article 标签
                document.querySelectorAll('article').forEach(function(a) {
                    var t = a.innerText ? a.innerText.trim() : '';
                    if (t.length > 10) parts.push(t);
                });

                // 3. 获取所有可见的较长文本块
                var seen = {};
                document.body.querySelectorAll('div, p, span, article, section, h1, h2, h3').forEach(function(el) {
                    if (el.offsetParent === null) return;
                    if (el.querySelector('div, p, article')) return; // skip containers
                    var t = el.innerText ? el.innerText.trim() : '';
                    if (t.length > 15 && t.length < 3000 && !seen[t]) {
                        seen[t] = true;
                        parts.push(t);
                    }
                });

                return parts.join('\\n\\n');
            ''')
            print(f"[MultimodalExtractor] 笔记文本: {len(text)} 字")

            # 截图
            screenshot = driver.get_screenshot_as_base64()

            # 检查是否有图片轮播，尝试切换获取更多截图
            images = []
            try:
                # 找下一张按钮
                next_btn = driver.execute_script('''
                    var btns = document.querySelectorAll('[class*="arrow"], [class*="next"], [class*="right"]');
                    return btns.length;
                ''')
                if next_btn > 0:
                    # 截取多张图
                    for i in range(min(3, next_btn)):
                        try:
                            driver.execute_script('''
                                var btn = document.querySelector('[class*="arrow"], [class*="next"], [class*="right"]');
                                if (btn) btn.click();
                            ''')
                            time.sleep(1)
                            images.append(driver.get_screenshot_as_base64())
                        except Exception:
                            pass
            except Exception:
                pass

            driver.quit()
            driver = None

            if not text or len(text.strip()) < 10:
                print("[MultimodalExtractor] 笔记页面未提取到有效文本")
                return None

            print(f"[MultimodalExtractor] 笔记文本: {len(text)} 字")

            # 发送给 Ollama 分析（文本 + 截图）
            focus_instruction = ""
            if hasattr(self, '_focus') and self._focus:
                focus_instruction = f"""
**【重要】只提取与「{self._focus}」相关的内容，忽略页面侧边栏推荐的其他游戏或无关话题。**
"""

            prompt = f"""请仔细阅读以下抖音图文笔记的全部内容，只提取其中与游戏相关的信息。{focus_instruction}
【笔记原文】
{text[:5000]}

请尽可能详尽地提取以下每一类信息（如果有的话）：
- 涉及的游戏名称
- 具体的活动/事件名称、类型（联动/更新/福利等）
- 活动时间、地点、参与方式
- 涉及的英雄、角色、皮肤、道具
- 活动规则、奖励内容、获取条件
- 玩家反馈和评价倾向
- 任何其他可能有用的细节

请用中文逐条列出。"""

            all_images = [screenshot] + images
            return self._call_ollama_vision(prompt, all_images[:4])

        except Exception as e:
            print(f"[MultimodalExtractor] 笔记提取失败: {e}")
            if driver:
                try: driver.quit()
                except: pass
            return None

    @staticmethod
    def _is_douyin_url(url: str) -> bool:
        """判断是否为抖音链接"""
        return any(d in url for d in ['douyin.com', 'v.douyin.com', 'tiktok.com'])

    @classmethod
    def _normalize_douyin_url(cls, url: str) -> str:
        """
        将各种抖音 URL 格式统一转换为 /video/{id} 格式
        """
        import re

        if re.search(r'/video/\d+', url):
            return url

        if 'v.douyin.com' in url:
            # 短链接先展开
            resolved = cls._resolve_short_link(url)
            if resolved and resolved != url:
                print(f"[MultimodalExtractor] 短链接展开: {resolved[:80]}...")
                return resolved
            return url

        # modal_id= 参数（搜索页/用户页弹窗视频）
        m = re.search(r'modal_id=(\d+)', url)
        if m:
            video_id = m.group(1)
            return f'https://www.douyin.com/video/{video_id}'

        return url

    # ============================================================
    # 微博视频提取
    # ============================================================

    @staticmethod
    def _is_weibo_url(url: str) -> bool:
        """判断是否为微博视频链接（非CDN直链）"""
        weibo_domains = ['weibo.com', 'weibo.cn', 'm.weibo.cn', 'video.weibo.com', 't.cn']
        return any(d in url for d in weibo_domains)

    def _download_weibo(self, url: str) -> Optional[str]:
        """
        从微博链接提取并下载视频

        流程：
        1. 从 URL 提取帖子 ID
        2. 调用微博移动端 API 获取视频信息（JSON）
        3. 从响应中解析视频下载地址
        4. 下载视频到临时文件
        """
        import re
        import tempfile

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://m.weibo.cn/',
            'Accept': 'application/json, text/plain, */*',
        }

        post_id = None

        # 模式1: ?layerid= 参数（微博个人主页视频）
        m = re.search(r'layerid=([A-Za-z0-9]+)', url)
        if m:
            post_id = m.group(1)
            print(f"[MultimodalExtractor] 微博 layerid 链接, ID: {post_id}")

        # 模式2: m.weibo.cn/detail/{id} 或 m.weibo.cn/status/{id}
        if not post_id:
            m = re.search(r'm\.weibo\.(?:cn|com)/(?:detail|status(?:es)?)/([A-Za-z0-9]+)', url)
            if m:
                post_id = m.group(1)
                print(f"[MultimodalExtractor] 微博移动端链接, ID: {post_id}")

        # 模式3: weibo.com/{uid}/{post_id}
        if not post_id:
            m = re.search(r'weibo\.com/\d+/([A-Za-z0-9]+)', url)
            if m:
                post_id = m.group(1)
                print(f"[MultimodalExtractor] 微博桌面链接, ID: {post_id}")

        # 模式4: video.weibo.com/show?fid={fid}
        if not post_id:
            m = re.search(r'fid=([A-Za-z0-9]+)', url)
            if m:
                post_id = m.group(1)
                print(f"[MultimodalExtractor] 微博视频链接, FID: {post_id}")

        # 模式4: t.cn 短链接 → 先展开
        if not post_id and 't.cn' in url:
            try:
                print(f"[MultimodalExtractor] 展开 t.cn 短链接...")
                resp = requests.head(url, headers=headers, allow_redirects=True, timeout=15)
                expanded = resp.url
                print(f"[MultimodalExtractor] 展开后: {expanded}")
                # 递归处理展开后的 URL
                return self._download_weibo(expanded)
            except Exception as e:
                print(f"[MultimodalExtractor] 短链接展开失败: {e}")
                return None

        if not post_id:
            print(f"[MultimodalExtractor] 无法从微博链接提取 ID: {url}")
            return None

        # 调用微博移动端 API 获取视频信息（带 Cookie 获取）
        video_url = None
        try:
            # 先访问微博首页获取 cookie（部分 API 需要）
            session = requests.Session()
            session.headers.update(headers)
            try:
                session.get('https://m.weibo.cn/', timeout=10)
            except Exception:
                pass  # cookie 获取失败不阻塞

            api_url = f'https://m.weibo.cn/statuses/show?id={post_id}'
            print(f"[MultimodalExtractor] 调用微博 API: {api_url}")
            resp = session.get(api_url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if data.get('ok') != 1:
                print(f"[MultimodalExtractor] 微博 API 返回 ok!=1: {data.get('msg', '未知错误')}")
                # 不直接 return None，继续尝试网页版

            # 从 page_info 中提取视频 URL
            page_info = data.get('data', {})
            if data.get('ok') == 1:
                # 可能是转发微博，尝试从 retweeted_status 中获取
                if not page_info.get('page_info'):
                    retweeted = page_info.get('retweeted_status', {})
                    page_info = retweeted

            media_info = page_info.get('page_info', {}).get('media_info', {})
            if not media_info:
                urls = page_info.get('page_info', {}).get('urls', {})
                video_url = (urls.get('mp4_720p_mp4') or urls.get('mp4_hd_mp4') or
                           urls.get('mp4_ld_mp4') or urls.get('mp4_720p') or urls.get('mp4_hd'))
            else:
                video_url = (media_info.get('stream_url_hd') or
                           media_info.get('stream_url') or
                           media_info.get('mp4_720p_mp4') or
                           media_info.get('mp4_hd_url') or
                           media_info.get('mp4_sd_url'))

            if not video_url:
                # 在完整响应中搜索 mp4 地址
                import json
                raw = json.dumps(data)
                url_match = re.search(r'(https?://[^"\']+?\.mp4[^"\']*?)', raw)
                if url_match:
                    video_url = url_match.group(1).replace('\\/', '/')

            if video_url:
                print(f"[MultimodalExtractor] 获取到视频地址: {video_url[:80]}...")

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else '?'
            print(f"[MultimodalExtractor] 微博 API 返回 HTTP {status_code}, 尝试网页版解析")
        except Exception as e:
            print(f"[MultimodalExtractor] 微博 API 调用失败: {e}, 尝试网页版解析")

        # API 失败 → 网页版解析
        if not video_url:
            video_url = self._extract_weibo_video_from_page(url, headers)

        if not video_url:
            print("[MultimodalExtractor] 未能从微博提取到视频地址")
            return None

        # 下载视频到临时文件
        return self._download_weibo_video(video_url)

    def _extract_weibo_video_from_page(self, url: str, headers: dict) -> Optional[str]:
        """从微博网页版页面源码中提取视频地址（API 不可用时的回退方案）"""
        import re

        try:
            page_headers = {**headers, 'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}

            # 同时尝试桌面版和移动版的页面
            urls_to_try = [url]
            # 如果是桌面版链接，也尝试对应的移动版
            if 'weibo.com' in url and 'm.weibo' not in url:
                m = re.search(r'(?:layerid|status(?:es)?/)?([A-Za-z0-9]+)$', url.split('?')[0])
                if not m:
                    m = re.search(r'layerid=([A-Za-z0-9]+)', url)
                if m:
                    urls_to_try.append(f'https://m.weibo.cn/status/{m.group(1)}')

            for page_url in urls_to_try:
                try:
                    resp = requests.get(page_url, headers=page_headers, timeout=15)
                    resp.raise_for_status()
                    html = resp.text

                    # 方法1：搜索 embedded JSON 数据（$render_data 或 __INITIAL_STATE__）
                    json_patterns = [
                        r'window\.\$render_data\s*=\s*(\{.+?\})\s*</script>',
                        r'window\.__INITIAL_STATE__\s*=\s*(\{.+?\});',
                        r'"page_info"\s*:\s*(\{.+?"urls"\s*:\s*\{.+?\}\})',
                    ]
                    for jp in json_patterns:
                        json_matches = re.findall(jp, html, re.DOTALL)
                        for jm in json_matches:
                            mp4_match = re.search(r'(https?://[^"\']+?\.mp4[^"\']*?)', jm)
                            if mp4_match:
                                video_url = mp4_match.group(1).replace('\\/', '/')
                                print(f"[MultimodalExtractor] 从JSON嵌入数据提取到: {video_url[:80]}...")
                                return video_url

                    # 方法2：搜索常见的视频 URL 模式
                    patterns = [
                        r'video-src\s*=\s*["\']([^"\']+)["\']',
                        r'stream_url["\']?\s*:\s*["\']([^"\']+)["\']',
                        r'(https?://[^"\'\s]+?\.mp4[^"\'\s]*)',
                        r'mp4_720p["\']?\s*:\s*["\']([^"\']+)["\']',
                        r'mp4_hd["\']?\s*:\s*["\']([^"\']+)["\']',
                    ]

                    for pattern in patterns:
                        matches = re.findall(pattern, html)
                        if matches:
                            video_url = matches[0].replace('\\/', '/')
                            print(f"[MultimodalExtractor] 网页版提取到视频地址: {video_url[:80]}...")
                            return video_url

                except Exception:
                    continue  # 尝试下一个 URL

            print("[MultimodalExtractor] 网页版未找到视频地址")
            return None
        except Exception as e:
            print(f"[MultimodalExtractor] 网页版提取失败: {e}")
            return None

    def _download_weibo_video(self, video_url: str) -> Optional[str]:
        """下载微博视频到临时文件"""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='_ytdl', delete=False)
        tmp.close()

        try:
            print(f"[MultimodalExtractor] 下载微博视频: {video_url[:80]}...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://weibo.com/',
            }
            resp = requests.get(video_url, headers=headers, stream=True, timeout=120)
            resp.raise_for_status()

            total = 0
            max_bytes = 200 * 1024 * 1024
            with open(tmp.name, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        break

            print(f"[MultimodalExtractor] 微博视频下载完成: {total / 1024 / 1024:.1f} MB（临时文件）")
            return tmp.name
        except Exception as e:
            print(f"[MultimodalExtractor] 微博视频下载失败: {e}")
            self._cleanup_temp(tmp.name)
            return None

    def _download_direct(self, url: str) -> Optional[str]:
        """流式下载直链视频，限制大小，处理完即删"""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='_ytdl', delete=False)
        tmp.close()

        try:
            print(f"[MultimodalExtractor] 流式下载直链: {url[:80]}...")
            resp = requests.get(url, stream=True, timeout=300, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp.raise_for_status()

            max_bytes = 300 * 1024 * 1024  # 300MB 上限
            total = 0
            with open(tmp.name, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        print(f"[MultimodalExtractor] 视频过大(>{max_bytes//1024//1024}MB)，截断下载")
                        break

            print(f"[MultimodalExtractor] 下载完成: {total / 1024 / 1024:.1f} MB（临时文件，处理完即删）")
            return tmp.name
        except Exception as e:
            print(f"[MultimodalExtractor] 直链下载失败: {e}")
            self._cleanup_temp(tmp.name)
            return None

    def _get_fresh_cookies(self, url: str) -> Optional[str]:
        """
        自动获取平台 Cookie，按优先级尝试：

        1. 需要登录态的平台（抖音/微博等）→ CDP 从浏览器获取真实 Cookie
        2. 其他平台 → requests.Session 访问首页获取基础 Cookie
        """
        domain = urlparse(url).netloc.lower()

        # 需要真实浏览器 Cookie 的平台
        # 抖音由 Selenium 单独处理，这里只处理微博
        needs_browser = any(d in domain for d in ['weibo.com', 'weibo.cn', 't.cn'])

        if needs_browser:
            result = self._get_cookies_via_cdp(url)
            if result:
                return result
            print("[MultimodalExtractor] CDP Cookie 获取失败，回退到基础 Cookie")

        # 普通平台：requests 访问首页获取基础 Cookie
        return self._get_cookies_via_requests(url)

    def _get_cookies_via_requests(self, url: str) -> Optional[str]:
        """通过 requests.Session 访问首页获取基础 Cookie"""
        from urllib.parse import urlparse
        import tempfile

        domain = urlparse(url).netloc

        if 'douyin.com' in domain:
            urls_to_visit = ['https://www.douyin.com/', url]
        elif 'bilibili.com' in domain:
            urls_to_visit = ['https://www.bilibili.com/', url]
        elif 'weibo.com' in domain or 'weibo.cn' in domain:
            urls_to_visit = ['https://m.weibo.cn/', url]
        else:
            parts = domain.split('.')
            root = '.'.join(parts[-2:]) if len(parts) >= 2 else domain
            urls_to_visit = [f'https://www.{root}/', url]

        try:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'zh-CN,zh;q=0.9',
            })
            for visit_url in urls_to_visit:
                try:
                    session.get(visit_url, timeout=15)
                except Exception:
                    pass

            cookies = dict(session.cookies)
            if not cookies:
                return None

            print(f"[MultimodalExtractor] requests 获取到 {len(cookies)} 个 Cookie: {list(cookies.keys())}")
            return self._save_cookies_netscape(cookies, domain)
        except Exception as e:
            print(f"[MultimodalExtractor] requests Cookie 获取失败: {e}")
            return None

    def _get_cookies_via_cdp(self, url: str) -> Optional[str]:
        """
        通过 Chrome DevTools Protocol 从浏览器获取真实 Cookie

        启动 headless Chrome（使用用户现有配置文件），通过 CDP WebSocket
        调用 Network.getCookies 获取完整的浏览器 Cookie（无需 DPAPI 解密）。
        """
        from urllib.parse import urlparse
        import tempfile
        import socket

        domain = urlparse(url).netloc
        # 提取根域名用于 Cookie 过滤
        parts = domain.split('.')
        root_domain = '.'.join(parts[-2:]) if len(parts) >= 2 else domain

        # 查找 Chrome
        chrome_paths = [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'),
        ]
        chrome = None
        for p in chrome_paths:
            if os.path.exists(p):
                chrome = p
                break
        if not chrome:
            print("[MultimodalExtractor] 未找到 Chrome")
            return None

        # 找可用端口
        def _find_port():
            for p in range(9222, 9250):
                try:
                    s = socket.socket()
                    s.bind(('127.0.0.1', p))
                    s.close()
                    return p
                except Exception:
                    continue
            return 9222

        port = _find_port()
        user_data = os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\User Data')

        print(f"[MultimodalExtractor] 启动 Chrome CDP (端口 {port})...")

        try:
            # 启动 headless Chrome
            proc = subprocess.Popen([
                chrome,
                f'--remote-debugging-port={port}',
                '--headless=new',
                f'--user-data-dir={user_data}',
                '--profile-directory=Default',
                '--disable-gpu',
                '--no-sandbox',
                '--disable-extensions',
                'about:blank'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(2)  # 等 Chrome 启动

            # 获取 WebSocket URL
            resp = requests.get(f'http://127.0.0.1:{port}/json', timeout=10)
            pages = resp.json()
            ws_url = None
            for page in pages:
                if page.get('type') == 'page':
                    ws_url = page.get('webSocketDebuggerUrl')
                    break

            if not ws_url:
                print("[MultimodalExtractor] 未找到 CDP WebSocket")
                proc.terminate()
                return None

            # 连接 WebSocket
            import websocket
            ws = websocket.create_connection(ws_url, timeout=15)

            def cdp(method, params=None):
                msg = json.dumps({'id': 1, 'method': method, 'params': params or {}})
                ws.send(msg)
                resp = json.loads(ws.recv())
                return resp.get('result', {})

            # 导航到首页获取该域名的 Cookie
            home = f'https://{domain}/' if not domain.startswith('www.') else f'https://{domain}/'
            cdp('Page.enable')
            cdp('Network.enable')
            cdp('Page.navigate', {'url': home})

            # 等待页面加载
            time.sleep(3)

            # 获取该域名的所有 Cookie
            result = cdp('Network.getCookies')
            all_cookies = result.get('cookies', [])

            # 过滤当前域名相关的 cookie
            cookies = {}
            for c in all_cookies:
                c_domain = c.get('domain', '')
                if root_domain in c_domain or domain in c_domain or c_domain in domain:
                    cookies[c['name']] = c['value']

            ws.close()
            proc.terminate()

            if not cookies:
                print("[MultimodalExtractor] CDP 未获取到 Cookie")
                return None

            print(f"[MultimodalExtractor] CDP 获取到 {len(cookies)} 个 Cookie: {list(cookies.keys())[:8]}")
            return self._save_cookies_netscape(cookies, domain)

        except ImportError:
            print("[MultimodalExtractor] websocket-client 未安装，无法使用 CDP")
            try:
                proc.terminate()
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"[MultimodalExtractor] CDP Cookie 获取失败: {e}")
            try:
                proc.terminate()
            except Exception:
                pass
            return None

    def _save_cookies_netscape(self, cookies: dict, domain: str) -> str:
        """将 cookie 字典保存为 Netscape 格式的临时文件"""
        import tempfile
        parts = domain.split('.')
        root_domain = '.' + '.'.join(parts[-2:]) if len(parts) >= 2 else domain

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        tmp.write('# Netscape HTTP Cookie File\n')
        tmp.write('# Auto-generated by MultimodalExtractor\n\n')
        for name, value in cookies.items():
            tmp.write(f'{root_domain}\tTRUE\t/\tFALSE\t0\t{name}\t{value}\n')
        tmp.close()
        return tmp.name

    def _download_with_ytdlp(self, url: str) -> Optional[str]:
        """
        使用 yt-dlp 按需下载平台视频

        自动获取基础 Cookie（无需登录或浏览器），解决抖音等平台的反爬限制。
        """
        import tempfile

        audio_tmp = tempfile.NamedTemporaryFile(suffix='_ytdl', delete=False)
        audio_tmp.close()

        cookie_file = None
        try:
            # 将 ffmpeg 目录加入 PATH（必须在 yt-dlp import 之前）
            ffmpeg_dir = self._get_ffmpeg_path()
            if os.path.exists(os.path.join(ffmpeg_dir, 'ffmpeg.exe')):
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

            from yt_dlp import YoutubeDL
            print(f"[MultimodalExtractor] yt-dlp 提取: {url[:80]}...")

            # Cookie 优先级：Selenium > 手动指定 > 自动获取基础 Cookie > 浏览器提取
            cookie_config = {}
            selenium_cookie = os.environ.get('_YTDLP_DOUYIN_COOKIE', '')
            manual_cookie = os.environ.get('YTDLP_COOKIE_FILE', '')
            if selenium_cookie and os.path.exists(selenium_cookie):
                cookie_config['cookiefile'] = selenium_cookie
                print(f"[MultimodalExtractor] 使用 Selenium Cookie: {selenium_cookie}")
            elif manual_cookie and os.path.exists(manual_cookie):
                cookie_config['cookiefile'] = manual_cookie
                print(f"[MultimodalExtractor] 使用手动 Cookie 文件: {manual_cookie}")
            else:
                # 自动获取平台基础 Cookie（无需登录）
                cookie_file = self._get_fresh_cookies(url)
                if cookie_file:
                    cookie_config['cookiefile'] = cookie_file
                    print(f"[MultimodalExtractor] 使用自动获取的 Cookie: {cookie_file}")
                else:
                    # 最后尝试浏览器 Cookie
                    for browser in ['chrome', 'edge', 'firefox']:
                        try:
                            cookie_config['cookiesfrombrowser'] = (browser,)
                            print(f"[MultimodalExtractor] 尝试从浏览器提取: {browser}")
                            break
                        except Exception:
                            continue

            ydl_opts = {
                'outtmpl': audio_tmp.name + '.%(ext)s',
                'format': 'bv*+ba/b',
                'ffmpeg_location': self._get_ffmpeg_path(),
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'force_generic_extractor': False,
                'socket_timeout': 30,
                'retries': 3,
                **cookie_config,
                # 不用 yt-dlp 的 FFmpegExtractAudio（它找不到 ffmpeg）
                # 我们自己的 _extract_from_video 会用 ffmpeg 提取音轨
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # yt-dlp 使用 .%(ext)s 后缀，查找实际下载的文件（跳过 0 字节占位文件）
            import glob
            base = audio_tmp.name
            candidates = [c for c in glob.glob(base + '*')
                         if not c.endswith('.txt') and os.path.getsize(c) > 0]
            actual_audio = candidates[0] if candidates else base

            if os.path.exists(actual_audio) and os.path.getsize(actual_audio) > 0:
                size_mb = os.path.getsize(actual_audio) / 1024 / 1024
                print(f"[MultimodalExtractor] 下载完成: {size_mb:.1f} MB（临时文件）")
                return actual_audio
            else:
                print("[MultimodalExtractor] 下载失败，回退到低分辨率")
                self._cleanup_temp(actual_audio)
                return self._download_video_lowres(url)

        except ImportError:
            print("[MultimodalExtractor] yt-dlp 未安装，仅支持直链")
            self._cleanup_temp(audio_tmp.name)
            return None
        except Exception as e:
            error_msg = str(e)
            print(f"[MultimodalExtractor] yt-dlp 下载失败: {error_msg}")
            if self._is_douyin_url(url):
                print("[MultimodalExtractor] 抖音提取失败。可尝试在 .env 中设置 YTDLP_COOKIE_FILE=浏览器导出的cookies.txt")
            elif self._is_weibo_url(url):
                print("[MultimodalExtractor] 微博提取失败。可尝试在 .env 中设置 YTDLP_COOKIE_FILE=浏览器导出的cookies.txt")
            self._cleanup_temp(audio_tmp.name)
            import glob
            for f in glob.glob(audio_tmp.name.rsplit('.', 1)[0] + '*'):
                self._cleanup_temp(f)
            if cookie_file:
                self._cleanup_temp(cookie_file)
            return self._download_video_lowres(url)
        finally:
            if cookie_file:
                self._cleanup_temp(cookie_file)

    def _download_video_lowres(self, url: str) -> Optional[str]:
        """回退方案：下载低分辨率（360p）视频用于抽帧"""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='_ytdl', delete=False)
        tmp.close()

        try:
            from yt_dlp import YoutubeDL
            print(f"[MultimodalExtractor] yt-dlp 下载低分辨率视频: {url[:80]}...")

            ydl_opts = {
                'outtmpl': tmp.name + '.%(ext)s',
                'format': 'worst[height>=240]/worst',
                'ffmpeg_location': self._get_ffmpeg_path(),
                'quiet': True,
                'no_warnings': True,
                'max_filesize': 100 * 1024 * 1024,
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            import glob
            actual = tmp.name
            candidates = glob.glob(tmp.name + '*')
            if candidates:
                actual = candidates[0]

            if os.path.exists(actual) and os.path.getsize(actual) > 0:
                print(f"[MultimodalExtractor] 低分辨率视频: {os.path.getsize(actual) / 1024 / 1024:.1f} MB（临时文件）")
                return actual
            return None
        except Exception as e:
            print(f"[MultimodalExtractor] 低分辨率视频下载失败: {e}")
            self._cleanup_temp(tmp.name)
            return None

    def _cleanup_temp(self, path: str):
        """清理临时文件"""
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass

    def _focus_instruction(self) -> str:
        """生成产品关注点过滤指令"""
        if hasattr(self, '_focus') and self._focus:
            return f"\n\n**【重要】只提取与「{self._focus}」直接相关的内容。** 忽略侧边栏推荐、其他游戏、无关话题。如果页面包含多款游戏的信息，只关注与「{self._focus}」相关的部分。"
        return ""

    # ============================================================
    # 图片处理
    # ============================================================

    def _extract_from_image(self, image_path: str) -> str:
        """提取单张图片信息"""
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return ""

        prompt = IMAGE_EXTRACTION_PROMPT + self._focus_instruction()
        result = self._call_ollama_vision(prompt, [image_b64])
        return result

    # ============================================================
    # 视频处理
    # ============================================================

    def _extract_from_video(self, video_path: str) -> str:
        """
        视频/音频处理：音频转录 + 场景切换抽帧 → Ollama 综合理解

        对于纯音频文件（从链接仅下载音轨的情况），跳过抽帧，仅转录。
        """
        # 检测是否为纯音频文件（从链接仅下载音轨）
        audio_exts = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.opus'}
        is_audio_only = Path(video_path).suffix.lower() in audio_exts

        # 路线 A：音频转录
        transcript = ""
        if is_audio_only:
            # 已经是音频文件，直接转录
            try:
                transcript = self._transcribe_audio(video_path)
            except Exception as e:
                print(f"[MultimodalExtractor] 音频转录失败: {e}")
        else:
            # 从视频中提取音轨再转录
            try:
                audio_path = self._extract_audio(video_path)
                if audio_path:
                    transcript = self._transcribe_audio(audio_path)
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[MultimodalExtractor] 音轨提取/转录失败: {e}")

        if transcript:
            print(f"[MultimodalExtractor] 转录文本: {transcript[:100]}...")
        else:
            print("[MultimodalExtractor] 未获取到音频内容")

        # 路线 B：场景切换抽帧（仅视频文件）
        frame_b64_list = []
        if not is_audio_only:
            try:
                frame_indices = self._detect_scene_changes(video_path)
                if frame_indices:
                    frame_b64_list = self._extract_frames_as_base64(video_path, frame_indices)
                    print(f"[MultimodalExtractor] 提取了 {len(frame_b64_list)} 个关键帧")
            except Exception as e:
                print(f"[MultimodalExtractor] 场景抽帧失败: {e}")

        # 综合
        if not transcript and not frame_b64_list:
            print("[MultimodalExtractor] 未能提取任何信息")
            return ""

        if transcript:
            prompt = VIDEO_EXTRACTION_PROMPT.format(transcript=transcript[:3000]) + self._focus_instruction()
        else:
            prompt = IMAGE_EXTRACTION_PROMPT + self._focus_instruction()

        self._unload_whisper()  # 释放 Whisper 模型节省内存

        if frame_b64_list:
            return self._call_ollama_vision(prompt, frame_b64_list)
        elif transcript:
            return self._summarize_transcript(transcript)
        else:
            return ""

    def _summarize_transcript(self, transcript: str) -> str:
        """仅有音频无画面时，用 LLM 做纯文本摘要"""
        focus = self._focus_instruction()
        prompt = f"""以下是视频的语音转录文本，请提取其中与游戏相关的话题和关键事件信息：{focus}
重点关注：
- 讨论的具体游戏话题
- 涉及的英雄、皮肤、活动、版本更新
- 玩家/主播关注的核心争议点或评价
- 提及的重要数据/时间节点

转录文本：
{transcript[:3000]}

请用简洁的中文列出提取到的关键信息，每条一行。"""

        try:
            url = f"{self.base_url.rstrip('/')}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[MultimodalExtractor] 转录摘要失败: {e}")
            return f"视频语音内容摘要：\n{transcript[:1500]}"

    # ============================================================
    # 场景检测
    # ============================================================

    def _detect_scene_changes(
        self,
        video_path: str,
        min_interval_sec: float = 2.0,
        max_frames: int = 15
    ) -> List[int]:
        """
        基于帧间直方图差异检测场景切换点

        Args:
            video_path: 视频路径
            min_interval_sec: 关键帧最小间隔（秒）
            max_frames: 最大关键帧数量

        Returns:
            关键帧的帧索引列表
        """
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[MultimodalExtractor] 无法打开视频")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # fallback
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_frame_interval = int(fps * min_interval_sec)

        keyframe_indices = []
        prev_hist = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每 0.5 秒检查一次（减少计算量）
            if frame_idx % max(1, int(fps * 0.5)) != 0:
                frame_idx += 1
                continue

            # 缩小帧以加速直方图计算
            small = cv2.resize(frame, (128, 72))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

                # 差异超过阈值 + 距离上一次关键帧足够远 → 记录
                if diff > 1.5:
                    if not keyframe_indices or (frame_idx - keyframe_indices[-1] >= min_frame_interval):
                        keyframe_indices.append(frame_idx)

            prev_hist = hist
            frame_idx += 1

            if len(keyframe_indices) >= max_frames:
                break

        cap.release()

        # 如果检测到的帧太少，补充首帧和末帧
        if len(keyframe_indices) < 2 and total_frames > 0:
            if 0 not in keyframe_indices:
                keyframe_indices.insert(0, 0)
            last_idx = total_frames - 1
            if last_idx not in keyframe_indices and last_idx > 0:
                keyframe_indices.append(last_idx)

        return keyframe_indices

    def _extract_frames_as_base64(self, video_path: str, frame_indices: List[int]) -> List[str]:
        """根据帧索引提取帧并编码为 base64"""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frames_b64 = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 压缩帧以控制大小：最大宽度 800px
                h, w = frame.shape[:2]
                if w > 800:
                    scale = 800 / w
                    frame = cv2.resize(frame, (800, int(h * scale)))

                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64 = base64.b64encode(buf).decode('utf-8')
                frames_b64.append(b64)

        cap.release()
        return frames_b64

    # ============================================================
    # 音频提取与转录
    # ============================================================

    def _get_ffmpeg_path(self) -> str:
        """获取 ffmpeg 可执行文件路径"""
        # 优先使用 venv 中的
        candidates = [
            os.path.join(os.path.dirname(sys.executable), 'ffmpeg.exe'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'commenter', 'Scripts', 'ffmpeg.exe'),
            'ffmpeg',
        ]
        for p in candidates:
            if os.path.exists(p) or p == 'ffmpeg':
                return p
        return 'ffmpeg'

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """从视频中提取音轨为 16kHz mono WAV"""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()

            ffmpeg = self._get_ffmpeg_path()
            cmd = [
                ffmpeg, '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-loglevel', 'error',
                tmp.name
            ]
            subprocess.run(cmd, check=True, timeout=120)
            return tmp.name
        except FileNotFoundError:
            print("[MultimodalExtractor] ffmpeg 未安装，跳过音频提取")
            return None
        except subprocess.TimeoutExpired:
            print("[MultimodalExtractor] ffmpeg 超时")
            return None
        except Exception as e:
            print(f"[MultimodalExtractor] 音轨提取失败: {e}")
            return None

    def _transcribe_audio(self, audio_path: str) -> str:
        """使用 faster-whisper 转录音频"""
        if not os.path.exists(audio_path):
            return ""

        try:
            model = self._get_whisper_model()
            segments, info = model.transcribe(audio_path, language='zh', beam_size=5)
            segments = list(segments)
            transcript = ' '.join(s.text.strip() for s in segments if s.text)
            print(f"[MultimodalExtractor] 转录完成: {len(transcript)} 字, 语言: {info.language}")
            return transcript
        except ImportError:
            print("[MultimodalExtractor] faster-whisper 未安装，跳过转录")
            return ""
        except Exception as e:
            print(f"[MultimodalExtractor] 转录失败: {e}")
            return ""

    def _get_whisper_model(self):
        """延迟加载 Whisper 模型（首次调用时下载）"""
        if self._whisper is None:
            from faster_whisper import WhisperModel
            print(f"[MultimodalExtractor] 加载 Whisper 模型: {self.whisper_model_size}")
            # 使用 int8 量化减少内存占用
            self._whisper = WhisperModel(
                self.whisper_model_size,
                device='cpu',
                compute_type='int8'
            )
        return self._whisper

    def _unload_whisper(self):
        """释放 Whisper 模型以节省内存"""
        if self._whisper is not None:
            print("[MultimodalExtractor] 释放 Whisper 模型")
            self._whisper = None
            import gc
            gc.collect()

    # ============================================================
    # Ollama API 调用
    # ============================================================

    def _call_ollama_vision(
        self,
        prompt: str,
        image_base64_list: List[str],
        timeout: int = 120
    ) -> str:
        """
        调用 Ollama 多模态 API

        Args:
            prompt: 文本 prompt
            image_base64_list: base64 编码的图片列表
            timeout: 超时秒数

        Returns:
            模型响应文本
        """
        url = f"{self.base_url.rstrip('/')}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_base64_list
                }
            ],
            "stream": False
        }

        try:
            print(f"[MultimodalExtractor] 调用 Ollama: {self.model}, "
                  f"{len(image_base64_list)} 张图片, prompt {len(prompt)} 字")
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("message", {}).get("content", "")
            elapsed = time.time() - t0
            print(f"[MultimodalExtractor] Ollama 响应: {len(result)} 字, 耗时 {elapsed:.1f}s")
            return result.strip()
        except requests.exceptions.Timeout:
            print(f"[MultimodalExtractor] Ollama 调用超时 ({timeout}s)")
            return ""
        except requests.exceptions.ConnectionError:
            print(f"[MultimodalExtractor] 无法连接到 Ollama: {self.base_url}")
            return ""
        except Exception as e:
            print(f"[MultimodalExtractor] Ollama 调用失败: {e}")
            return ""

    # ============================================================
    # 工具方法
    # ============================================================

    def _encode_image(self, image_path: str, max_width: int = 1200) -> Optional[str]:
        """读取图片并编码为 base64，过大的图片先缩放"""
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                print(f"[MultimodalExtractor] 无法读取图片: {image_path}")
                return None

            h, w = img.shape[:2]
            if w > max_width:
                scale = max_width / w
                img = cv2.resize(img, (max_width, int(h * scale)))

            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode('utf-8')
        except ImportError:
            # OpenCV 不可用时使用纯 Python 回退
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')


# ============================================================
# 便捷函数
# ============================================================

def create_extractor() -> Optional[MultimodalExtractor]:
    """创建提取器（检查 Ollama 是否可用）"""
    if not OLLAMA_ENABLED:
        print("[MultimodalExtractor] OLLAMA_ENABLED=false，跳过初始化")
        return None

    try:
        # 快速检查 Ollama 是否可达
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            print(f"[MultimodalExtractor] Ollama 连接成功: {OLLAMA_BASE_URL}")
            return MultimodalExtractor()
        else:
            print(f"[MultimodalExtractor] Ollama 响应异常: {resp.status_code}")
            return None
    except Exception as e:
        print(f"[MultimodalExtractor] Ollama 不可用: {e}")
        return None


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) < 2:
        print("用法: python multimodal_extractor.py <图片/视频路径>")
        sys.exit(1)

    extractor = MultimodalExtractor()
    result = extractor.extract(sys.argv[1])
    print(f"\n提取结果:\n{result}")
