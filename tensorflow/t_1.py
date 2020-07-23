import tensorflow as tf

'''
Tensorflow흐름
1. 그래프 생성 -> 변수 및 계산할 식 생성
2. session생성 및 session.run(그래프)- > 값을 연산하는다.
3. 파이썬으로 값 리턴
'''

x = tf.constant(1) #tensorflow 타입의 변수 생성(tensor) 및 값 할당
y = tf.constant(2)
z = tf.constant(3)

a = x*y+z #연산식 (그래프 빌딩)

sess = tf.compat.v1.Session() #Tensorflw Session 생성

b = sess.run(a) #Session 실행

print (b)


#텐서 생성(행열) 벡터는 1차원 행열은 2차원
a = tf.constant([[1,2],[2,3],[5,6]]) #numpy이와 같이 가장 큰 자료형의 타입으로 통일된다.
print(a.dtype)
print(a.shape)
print(sess.run(tf.rank(a)))

##텐서의 재구성 행열 바꾸기
##원소의 개수가 같아야 한다.
a = tf.constant([[1,2],[2,3],[5,6],[5,6],[5,6],[5,6]])
b = tf.reshape(a,[3,4])#행과 열의 재구성
c = tf.transpose(b)#행과 열을 교환
c = a * b #2벡터를 곲한값이 들어 간다.
c = tf.equal(a,b)#행열을 비교 후 크기가 같은 행열의 tf타입의 변수에 할당
print(sess.run(b))
print(sess.run(c))


#####자료형
'''
1. tf.constant : 상수
 -초기화 되면서 메모리 크기와 타입이 결정
 - 세선 실행 시 자동으로 초기화
 - 함수의 원형 : tf.constant(value , dtype,shape)dtype, shape 생략가능 -> 첫번째 데이터를 보고 유추함
2. tf.placeholder : 학습 데이터를 저장하기 위한 가변크기 메모리
 - 학습및 테스트 데이터를 담는 그릇역활
 - 자료구조만 정해주고 나중에 데이터를 공급받음
  -반복적으로 feeding가능
  -학습시 학습 데이터를 테스트시 테스트 데이터를 전달
  -th.placeholder(dtype,shape) dtype = float32 반드시 정의해야함
  ex) x = tf.placeholder(dtype = tf.float32)
      input_data = [1,2,3,4,5]
      sess.run(x,feed_dict={x:input_data})feed_dict로 inpu_data를 x에 대입 후 x를 실행
3. tf.Variable : 학습 파라미터를 저장하기 위한 고정크기 메모리
  - 반복적으로 읽고 쓸수 있는 기능
  - constant처럼 고정된 크기의 메모리 공간
  - 초기화가 받드시 되어야 한다*
  - 함수 원형 tf.Variable(value,dtype,shape)
inputdata=[1,2,3]
w = tf.Varible(inputdata,dtype = tf.float32)
init = tf.global_variables_initalizer()#tf.Variable변수를 초기화 시키기 위한 객체 생성
sess.run(init) # 초기화 실행
'''


#################
