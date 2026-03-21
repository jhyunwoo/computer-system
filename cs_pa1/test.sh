#!/bin/bash

# 입력과 출력 파일 번호 목록

make

for i in 1 2 3 4 5
do
  echo "=== Running test $i ==="

  # 프로그램 실행: examples/input{i}.txt → result/output{i}.txt
  ./main < examples/input$i.txt > result/output$i.txt

  # 결과 비교
  echo "--- diff output$i.txt ---"
  diff examples/output$i.txt result/output$i.txt
  echo ""
done