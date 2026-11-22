# Mercury Java Callback Interface

이 디렉토리는 Mercury ASR 시스템의 Java callback interface를 정의합니다.

## 파일 구조

- `DecoderCallback.java` - 메인 callback interface
- `DecoderCallbackExample.java` - 사용 예제 클래스
- `README.md` - 사용법 가이드

## 사용법

### 1. DecoderCallback Interface 구현

```java
import com.chisnova.mercury.DecoderCallback;

public class MyDecoderHandler implements DecoderCallback {
    
    @Override
    public void onResult(int[] ids, int size) {
        // ASR 결과 처리 로직
        System.out.println("Received " + size + " tokens");
        
        // 토큰 ID를 텍스트로 변환하거나 다른 처리
        for (int i = 0; i < size; i++) {
            System.out.print(ids[i] + " ");
        }
        System.out.println();
    }
}
```

### 2. JNI에서 Callback 설정

```java
// Java 측에서 callback 객체 생성
MyDecoderHandler handler = new MyDecoderHandler();

// Native C++ 코드에서 callback 설정
// (JNI wrapper를 통해 호출)
nativeSetCallback(handler);
```

### 3. Native C++ 측 JNI 코드

```cpp
// C++ JNI wrapper
extern "C" JNIEXPORT void JNICALL
Java_com_chisnova_mercury_AsrDecoder_setCallback(JNIEnv* env, jobject thiz, jobject callback) {
    auto java_callback = CallbackFactory::createJavaCallback(env, callback);
    decoder->setCallback(java_callback);
}
```

## 중요 사항

### 1. 메서드 이름 고정
- **반드시** `onResult` 메서드를 구현해야 합니다
- 메서드 이름을 변경하면 안 됩니다 (C++ 코드에서 하드코딩됨)

### 2. 메서드 시그니처
```java
void onResult(int[] ids, int size)
```
- `ids`: 토큰 ID 배열
- `size`: 토큰 개수
- 반환 타입: `void`

### 3. 패키지 구조
- 패키지: `com.chisnova.mercury`
- 디렉토리: `api/java/jni_wrapper/com/chisnova/mercury/`

## 예제 실행

```java
// 예제 callback 생성
DecoderCallbackExample callback = new DecoderCallbackExample("TestCallback");

// Native 코드에 callback 설정
// (실제 구현에서는 JNI wrapper를 통해 호출)
```

## 주의사항

1. **스레드 안전성**: Callback은 네이티브 스레드에서 호출될 수 있습니다
2. **메모리 관리**: JNI Global Reference가 자동으로 관리됩니다
3. **예외 처리**: 네이티브 코드에서 Java 예외가 적절히 처리됩니다
4. **성능**: 콜백 호출은 실시간으로 발생하므로 가벼운 처리를 권장합니다

## 확장

새로운 callback 기능이 필요한 경우:

1. `DecoderCallback` interface에 새로운 메서드 추가
2. C++ 코드에서 해당 메서드 호출 로직 추가
3. Java 구현체에서 새로운 메서드 구현 