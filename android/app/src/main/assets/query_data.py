import sqlite3

def query_conan_data():
    # 데이터베이스 파일 경로
    db_path = 'mercury.db'
    
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # SpeechContents 테이블에서 conan 데이터 조회
        cursor.execute("SELECT * FROM Contents WHERE CategoryName = 'conan'")
        
        # 결과 가져오기
        results = cursor.fetchall()
        
        # 컬럼 이름 가져오기
        column_names = [description[0] for description in cursor.description]
        
        # 결과 출력
        print(f"Found {len(results)} records for 'conan' category:")
        print("-" * 80)
        
        for row in results:
            for col_name, value in zip(column_names, row):
                print(f"{col_name}: {value}")
            print("-" * 80)
        
        # 연결 종료
        conn.close()
        
        return results
        
    except sqlite3.Error as e:
        print(f"SQLite 오류: {e}")
        return None
    except Exception as e:
        print(f"일반 오류: {e}")
        return None

if __name__ == "__main__":
    query_conan_data()
