# -*- coding: utf-8 -*-



from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select

options = Options()
options.chrome_executable_path='chromedriver.exe'

driver=webdriver.Chrome(options=options)
driver.get('https://bonexpert.com/ahp/')
           

# select = Select(driver.find_elements(By.CLASS_NAME,"form-control input-sm ng-valid ng-not-empty ng-dirty ng-valid-parse ng-touched"))
# select = driver.find_elements(By.ID,"ethnicity")

# select = select.select_by_value("AsiChi")

# ethnicity = driver.find_elements(By.ID,"ethnicity")
# ethnicity.send_keys('Asian Chinese')


select = Select(driver.find_element(By.ID,"ethnicity"))
select.select_by_index(8)

driver.find_element(By.ID,"boneage").send_keys("12.2")
driver.find_element(By.ID,"age").send_keys("13.1")
driver.find_element(By.ID,"height").send_keys("167")
driver.find_element(By.ID,"fheight").send_keys("189")
driver.find_element(By.ID,"mheight").send_keys("165")


driver.find_element(By.ID,"boneage").clear()
driver.find_element(By.ID,"age").clear()
driver.find_element(By.ID,"height").clear()
driver.find_element(By.ID,"fheight").clear()
driver.find_element(By.ID,"mheight").clear()

AHP = driver.find_element(By.CLASS_NAME,"col-xs-3 ng-binding")


AHP = driver.find_elements(By.ID,"ahp")





for ahp in AHP:

    print(ahp.text)


driver.close()




import requests
from bs4 import BeautifulSoup
hotpage = requests.get("https://bonexpert.com/ahp/#/?gdr=Male&eth=AsiChi&age=13.1&ba=12.2&h=167&fh=170&mh=160")

main = BeautifulSoup(hotpage.text, 'html.parser')

board_find = hotpage.find_all(class_='row')
















