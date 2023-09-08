import os
import re
from typing import *
import time
import uuid
import json
import base64
from selenium import webdriver
import requests

command_executor = os.getenv("COMMAND_EXECUTOR", "http://127.0.0.1:4444")


def get_file_content(driver, path):
    elem = driver.execute_script(
        "var input = window.document.createElement('INPUT'); "
        "input.setAttribute('type', 'file'); "
        "input.hidden = true; "
        "input.onchange = function (e) { e.stopPropagation() }; "
        "return window.document.documentElement.appendChild(input); ")
    elem._execute('sendKeysToElement', {'value': [path], 'text': path})
    result = driver.execute_async_script(
        "var input = arguments[0], callback = arguments[1]; "
        "var reader = new FileReader(); "
        "reader.onload = function (ev) { callback(reader.result) }; "
        "reader.onerror = function (ex) { callback(ex.message) }; "
        "reader.readAsDataURL(input.files[0]); "
        "input.remove(); "
        , elem)
    if not result.startswith('data:'):
        raise Exception("Failed to get file content: %s" % result)
    return base64.b64decode(result[result.find('base64,') + 7:])


class HtmlReader:
    def __init__(self, command_executor=command_executor, **kwargs):
        self.command_executor = command_executor
        self.__setup_driver__()

    def __setup_driver__(self):
        """
        set up selenium driver
        """
        self.chrome_options = webdriver.ChromeOptions()
        settings = {
            "recentDestinations": [{
                "id": "Save as PDF",
                "origin": "local",
                "account": "",
            }],
            "selectedDestinationId": "Save as PDF",
            "version": 2,
            "isHeaderFooterEnabled": False,
            # "customMargins": {},
            "isCssBackgroundEnabled": False,
            # calculated_print_options = {
            'landscape': True,
            'displayHeaderFooter': False,
            'printBackground': False,
            'preferCSSPageSize': True,
            "mediaSize": {"width_microns": 266.7, "height_microns": 420000}
            # 'pageSize': 'letter',
            # "paperWidth": 10.5,
            # "paperHeight": 15
            # }
        }
        prefs = {
            'printing.print_preview_sticky_settings.appState': json.dumps(settings),
            'savefile.default_directory': "/home/seluser/data"
        }
        self.chrome_options.add_experimental_option('prefs', prefs)
        self.chrome_options.add_argument('--kiosk-printing')
        self.chrome_options.add_argument('--enable-print-browser')
        self.chrome_options.add_argument("start-maximized")

    def __check_link__(self, url: str = None, retry=2):
        """
        check link if it already a file then download directly
        url: link to the website
        """
        if url:
            for _ in range(retry + 1):
                response = requests.get(url)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type")
                    if content_type:
                        if "pdf" in "".join(response.headers.get("content-type")):
                            file = response.content
                            return True, file
        return False, None

    def html_to_pdf(self, url: str = None, print_options: Dict = {}) -> bytes:
        """
        convert html to pdf
        url: link to the website
        content:
        print_option: more option or update option
        """
        check_pdf_file, pdf_content = self.__check_link__(url)
        if check_pdf_file:
            return pdf_content
        with webdriver.Remote(options=self.chrome_options, command_executor=self.command_executor) as driver:
            driver.get(url)
            time.sleep(2)
            # click agree term or cookies if appear
            js_script_click_accept_term = """
                const buttons = document.getElementsByTagName("button");
                for (let i = 0; i < buttons.length; i++) {
                    // Check if the button ID contains "accept" or "agree"
                    if (buttons[i].id.includes("accept") || buttons[i].id.includes("agree")) {
                        // Click the button
                        buttons[i].click();
                    }   
                }"""
            driver.execute_script(js_script_click_accept_term)
            driver.get(url)
            time.sleep(2)
            # delete all element whose tagname in list
            js_script_delete_element_follow_tag_name = """
                const elements = document.getElementsByTagName("tagName");
                const elementsArray = Array.from(elements);
                elementsArray.forEach(function(element) {
                element.remove();
                });"""
            remove_tag_name_token = ['footer', 'header', 'app-footer', 'HEADER', 'FOOTER']
            for token in remove_tag_name_token:
                remove_script = re.sub('tagName', token, js_script_delete_element_follow_tag_name)
                driver.execute_script(remove_script)

            # remove all element contain header in class name
            js_script_delete_element_follow_class_name = """
                const headerElements = document.body.querySelectorAll('[class*="__token__"]');
                headerElements.forEach((headerElement) => {
                headerElement.remove();
                });"""
            remove_class_name_token_list = ['footer-wrap', 'birdseye-header']
            for token in remove_class_name_token_list:
                remove_script = re.sub('__token__', token, js_script_delete_element_follow_class_name)
                driver.execute_script(remove_script)

            # remove all element contain id
            js_script_delete_element_follow_id = """
                    const headerElements = document.querySelectorAll("[id*='__token__']");
                    headerElements.forEach((element) => {
                    element.parentNode.removeChild(element);
                    });"""
            remove_id_token_list = ['app-footer', 'header', 'HEADER']
            for token in remove_id_token_list:
                remove_script = re.sub('__token__', token, js_script_delete_element_follow_id)
                driver.execute_script(remove_script)

            # delete all element whose class name in list
            js_script_delete_element_follow_exactly_class_name = """
                    const elements = document.getElementsByClassName("className");
                    const elementsArray = Array.from(elements);
                    elementsArray.forEach(function(element) {
                    element.remove();
                    });"""
            remove_class_name_token = ['footer', 'header', 'HEADER', 'FOOTER', ]
            for token in remove_class_name_token:
                remove_script = re.sub('className', token, js_script_delete_element_follow_exactly_class_name)
                driver.execute_script(remove_script)

            cachefile = "%s-%s" % ('test', uuid.uuid4().hex)
            title = driver.find_element('tag name', 'title')
            driver.execute_script(f"arguments[0].innerText = '{cachefile}'", title)
            driver.set_script_timeout(300)
            time.sleep(2)
            driver.execute_script('window.print();')
            time.sleep(1)
            return get_file_content(driver, f"/home/seluser/data/{cachefile}.pdf")
