## COVID-19 Project - IOT Project

- 주제 : Python + 라즈베리파이를 이용한 COVID-19 Care
- 사용 Program언어 : Python
- 목표 : 
    1. Raspberry를 이용한 구동 System
    2. Raspberry와 Opencv를 이용한 이미지 처리
    3. 구성 IOT System 과 상위 Application와의 통신 구현
  
- 주요 구성 :
    1. Program 구성 :
      + Kivy로 GUI 구성, Function구현(얼굴 인식, 열화상 측정, MP3재생등)<br/>
    
          <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/flow.JPG" width="500">

    2. Hardware 구성 :
      
          <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/hardwareflow.JPG" width="500">
        
    3. GUI 구성(라이브러리 : kivy) :
      + MP3듣기, 열측정
    
          <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/screen1.JPG" width="250">&nbsp&nbsp&nbsp    <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/screen2.JPG" width="250">      <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/screen3.JPG" width="250">  
        
- 실제 구현 & 기증(경산의 모 어린이집 기증) :

    <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/p1.jpg" width="150" height="250">   <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/p2.jpg" width="150" height="250">   <img src = "https://github.com/disco87/covid-19Project/blob/master/MD-images/people.JPG" width="150" height="250">


- 언론사 게재(영남일보) : [ 코로나 블루 대처를 위한 스마트 로봇 및 어플 개발](https://m.yeongnam.com/view.php?key=20200602001339189)

- 첫 개발 하면서 미흡한점 & 아쉬운 점
  + 상위 Application 부재로 인한 상위 데이터 통신 미구현
  + Motor 출력 부족으로 인한 불안정한 구동
  + 라즈베리파이 발열처리 부족으로 인한 System Down
  




