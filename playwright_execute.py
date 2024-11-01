from playwright.sync_api import sync_playwright
import logging
from datetime import datetime
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"


def click_text_on_page(url, text_to_click):
    with sync_playwright() as p:
        # Launch a browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to the target URL
        logging.info(f"Opening the URL: {url}")
        page.goto(url)

        # Use XPath to find the element containing the text
        logging.info(f"Searching for element containing text: '{text_to_click}'")
        element = page.locator(f"//*[contains(text(), '{text_to_click}')]")

        # Check if the element exists
        if element.count() > 0:
            logging.info(
                f"Element found with text: '{text_to_click}', attempting to click."
            )
            element.first.click()

            # Take a screenshot after clicking
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.png"
            page.screenshot(path=screenshot_path)
            logging.info(f"Screenshot taken and saved at: {screenshot_path}")
        else:
            logging.warning(f"No element found with text: '{text_to_click}'")

        # Close the browser
        logging.info("Closing the browser.")
        browser.close()


def capture_screenshot(url, screenshot_path="screenshot.png"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        logging.info(f"Opening URL: {url}")
        page.goto(url)
        page.screenshot(path=screenshot_path)
        logging.info(f"Screenshot saved at: {screenshot_path}")
        browser.close()


def generate_action_plan(extracted_text, instructions):
    pass


if __name__ == "__main__":
    url = "https://example.com"
    click_text_on_page(url=url, text_to_click="More information.")
