import sqlite3
import os


def insert_contents(uid_counter, topic_counter, level, topic_path, topic_dir, cursor, conn):
    # 영어 파일과 한국어 파일 찾기
    eng_file = os.path.join(topic_path, f"{topic_dir}.eng")
    kor_file = os.path.join(topic_path, f"{topic_dir}.kor")
    wlist_file = os.path.join(topic_path, f"{topic_dir}.wlist")
    
    if os.path.exists(eng_file) and os.path.exists(kor_file) and os.path.exists(wlist_file):
        print("is touch2")
        try:
            # 영어 파일 읽기
            with open(eng_file, 'r', encoding='utf-8') as f:
                eng_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # 한국어 파일 읽기
            with open(kor_file, 'r', encoding='utf-8') as f:
                kor_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # wlist 파일 읽기
            with open(wlist_file, 'r', encoding='utf-8') as f:
                wlist_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # 각 문장 쌍을 데이터베이스에 삽입
            for sentence_idx, (eng_sentence, kor_sentence, wlist) in enumerate(zip(eng_lines, kor_lines, wlist_lines)):
                print(wlist)
                opus_file = os.path.join(topic_path, wlist)
                audio_size = 0
                if os.path.exists(opus_file):
                    try:
                        import subprocess
                        # ffprobe를 사용하여 opus 파일의 길이를 초 단위로 가져오기
                        result = subprocess.run([
                            'ffprobe', '-v', 'quiet', '-show_entries', 
                            'format=duration', '-of', 'csv=p=0', opus_file
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0 and result.stdout.strip():
                            audio_size = int(float(result.stdout.strip()) * 1000)  # 밀리초로 변환
                    except Exception as audio_e:
                        print(f"오디오 길이 계산 오류 - {opus_file}: {audio_e}")
                        audio_size = 0
                cmd = f'''
                    INSERT INTO Contents (UID, CategoryType, ContentsID, SentenceID, CategoryName, EngLabel, KorLabel, LabelAudioSize)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.execute(cmd, (uid_counter, level, topic_counter, sentence_idx, topic_dir, eng_sentence, kor_sentence, audio_size))
                conn.commit()
                uid_counter += 1
        except Exception as e:
            print(f"파일 읽기 오류 - {topic_dir}: {e}")
        return uid_counter


def create_database():
    # 데이터베이스 파일 경로
    db_path = 'mercury.db'
    
    try:
        # 기존 데이터베이스 파일이 있으면 삭제
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"기존 데이터베이스 파일 '{db_path}' 삭제됨")
        
        # 데이터베이스 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("데이터베이스 연결 성공")
        
        # Contents Table 생성
        cursor.execute('''
            CREATE TABLE Contents (
                UID INTEGER PRIMARY KEY AUTOINCREMENT,
                CategoryType INTEGER,
                ContentsID INTEGER,
                SentenceID INTEGER,
                CategoryName TEXT,
                EngLabel TEXT,
                KorLabel TEXT,
                LabelAudioSize INTEGER
            )
        ''')
        print("Contents 테이블 생성 완료")

        # Attempt Table 생성
        cursor.execute('''
            CREATE TABLE Attempt (
                Date TEXT,
                AID TEXT,
                UID INTEGER,
                NumWord INTEGER,
                Transcript TEXT,
                RecordPath TEXT,
                Fluency REAL,
                Pronounce REAL
            )
        ''')
        print("Attempt 테이블 생성 완료")
        
        # Word History Table 생성
        cursor.execute('''
            CREATE TABLE WordHistory (
                Date TEXT,
                AID TEXT,
                WordIdx INTEGER,
                Word TEXT,
                NumPhone INTEGER,
                Color TEXT,
                Score REAL,
                Comment TEXT
            )
        ''')
        print("WordHistory 테이블 생성 완료")
        
        # Phone History Table 생성 (Data -> Date로 수정)
        cursor.execute('''
            CREATE TABLE PhoneHistory (
                Date TEXT,
                AID TEXT,
                WordIdx INTEGER,
                PhoneIdx INTEGER,
                Phone TEXT,
                Category TEXT,
                Score REAL
            )
        ''')
        print("PhoneHistory 테이블 생성 완료")
        
        # 변경사항 저장 및 연결 종료
        conn.commit()
        conn.close()

        # Contents 테이블에 데이터 삽입
        contents_path = "/Users/chisnova/mercury/android_mercury/app/src/main/assets/contents"
        
        if os.path.exists(contents_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            uid_counter = 1
            
            # contents 디렉토리 순회
            for level_dir in sorted(os.listdir(contents_path)):
                level_path = os.path.join(contents_path, level_dir)
                
                if os.path.isdir(level_path) and level_dir.startswith(('0.', '1.', '2.')):
                    level = int(level_dir.split('.')[0])
                    
                    # 각 레벨 디렉토리 내의 주제 디렉토리 순회
                    topic_counter = 1
                    for topic_dir in sorted(os.listdir(level_path)):
                        topic_path = os.path.join(level_path, topic_dir)
                        if "0" in level_dir:
                            if os.path.isdir(topic_path):
                                uid_counter = insert_contents(uid_counter, topic_counter, level, topic_path, topic_dir, cursor, conn)
                                topic_counter += 1
                        if "1" in level_dir:
                            theme = sorted(os.listdir(topic_path))
                            for theme_dir in theme:
                                theme_path = os.path.join(topic_path, theme_dir)
                                theme_name = topic_dir + "_" + theme_dir
                                print(theme_name)
                                print(theme_path)
                                if os.path.isdir(theme_path):
                                    print("is touch")
                                    uid_counter = insert_contents(uid_counter, topic_counter, level, theme_path, theme_dir, cursor, conn)
                                    topic_counter += 1
                        if "2" in level_dir:
                            if os.path.isdir(topic_path):
                                uid_counter = insert_contents(uid_counter, topic_counter, level, topic_path, topic_dir, cursor, conn)
                                topic_counter += 1

            
            # 변경사항 저장
            conn.commit()
            conn.close()
        else:
            print(f"Contents 경로를 찾을 수 없습니다: {contents_path}")
            
    except sqlite3.Error as e:
        print(f"SQLite 오류: {e}")
    except Exception as e:
        print(f"일반 오류: {e}")
    finally:
        # 연결이 열려있다면 닫기
        try:
            if conn:
                conn.close()
        except:
            pass

if __name__ == "__main__":
    create_database()
