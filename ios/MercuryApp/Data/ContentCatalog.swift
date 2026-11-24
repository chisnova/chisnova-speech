import Foundation

// 안드로이드 LessonData.kt / Level1Activity.kt / activity_level3.xml 에서 옮겨 온 표시용 메타데이터.
//
// 두 앱의 화면을 똑같이 맞추기 위해 목록과 순서를 안드로이드에 맞춘다. 그래서 빈출표현
// 정리는 폴더를 훑지 않고 여기 적힌 대로만 보여 준다 — 안드로이드에는 한 폴더를 두 항목이
// 가리키는 곳이 있고(감정 표현하기/의견 표현, 사과하기/예의있는 거절), 화면에 내보내지 않는
// 폴더도 있다(describing_locations, making_invitations). 폴더를 훑으면 그 구성이 깨진다.
enum ContentCatalog {

    struct Category {
        let id: String
        let title: String
        let emoji: String
        /// 안드로이드 drawable 이름 — IconPaths.table 의 열쇠
        let icon: String
        /// 안드로이드 bg_icon_* 에 대응하는 색 이름
        let tint: String
    }

    /// 화면에 나오는 주제 하나. 여러 항목이 같은 폴더를 가리킬 수 있어 id 와 folder 를 나눈다.
    struct Topic {
        /// 안드로이드 LessonData 의 레슨 id — 목록에서 고유하다
        let id: String
        let title: String
        let category: String
        /// 실제 콘텐츠 폴더(=파일 이름)
        let folder: String
        let icon: String
    }

    struct Speech {
        let folder: String
        let title: String
        let subtitle: String
        let image: String
    }

    /// 1.basic 분류 — 순서가 화면 순서다
    static let basicCategories: [Category] = [
        Category(id: "basic", title: "기초", emoji: "📚", icon: "ic_book_open_24", tint: "bg_icon_blue"),
        Category(id: "real_life", title: "실생활", emoji: "🏪", icon: "ic_coffee_24", tint: "bg_icon_green"),
        Category(id: "communication", title: "소통", emoji: "💬", icon: "ic_message_circle_24", tint: "bg_icon_blue"),
        Category(id: "important", title: "중요", emoji: "🚨", icon: "ic_alert_circle_24", tint: "bg_icon_red"),
    ]

    /// 1.basic 주제 — 안드로이드 Level1 화면과 같은 순서·이름
    static let basicTopics: [Topic] = [
        Topic(id: "greetings", title: "인사말과 기본 표현", category: "basic", folder: "greeting_basic_expression", icon: "ic_users_24"),
        Topic(id: "daily_conversation", title: "일상 대화 표현", category: "basic", folder: "phone_conversation", icon: "ic_people_chat_24"),
        Topic(id: "emotions", title: "감정 표현하기", category: "basic", folder: "expressing_opinions", icon: "ic_star_24"),
        Topic(id: "time_date", title: "시간과 날짜 표현", category: "basic", folder: "telling_time", icon: "ic_clock_24"),
        Topic(id: "food_ordering", title: "음식 주문하기", category: "real_life", folder: "ordering_food", icon: "ic_coffee_24"),
        Topic(id: "directions", title: "방향 안내 (Directions)", category: "real_life", folder: "asking_for_directions", icon: "ic_map_pin_24"),
        Topic(id: "asking_help", title: "도움 요청하기(Request Help)", category: "real_life", folder: "asking_for_help", icon: "ic_help_circle_24"),
        Topic(id: "shopping", title: "쇼핑하기 (Shopping & Ordering)", category: "real_life", folder: "shopping", icon: "ic_shopping_cart_24"),
        Topic(id: "apologies", title: "사과하기 (Apologies)", category: "communication", folder: "declining_politely", icon: "ic_volunteer_activism_24"),
        Topic(id: "making_plans", title: "계획 세우기(Making Plans)", category: "communication", folder: "making_appointment", icon: "ic_edit_calendar_24"),
        Topic(id: "opinions_feelings", title: "의견 표현 (Opinions & Feelings)", category: "communication", folder: "expressing_opinions", icon: "ic_thumbs_up_down_24"),
        Topic(id: "polite_disagreement", title: "예의있는 거절(Polite Disagreement)", category: "communication", folder: "declining_politely", icon: "ic_x_24"),
        Topic(id: "small_talk", title: "스몰토크(Small Talk)", category: "communication", folder: "introducing_people", icon: "ic_wb_cloudy_24"),
        Topic(id: "suggestions_advice", title: "제안하기(Suggestion & Advice)", category: "communication", folder: "making_suggestion", icon: "ic_lightbulb_24"),
        Topic(id: "emergency", title: "비상상황 대처 (Emergency)", category: "important", folder: "emergency_situation", icon: "ic_mic_24"),
    ]

    /// 주제별 콘텐츠 분류 폴더 -> 표시 정보
    static let articleCategories: [String: Category] = [
        "culture": Category(id: "culture", title: "문화 (Culture)", emoji: "🎨", icon: "ic_palette_24", tint: "bg_icon_green"),
        "economics": Category(id: "economics", title: "경제 (Economics)", emoji: "💰", icon: "ic_dollar_24", tint: "bg_icon_blue"),
        "entertain": Category(id: "entertain", title: "엔터테인먼트 (Entertainment)", emoji: "🎬", icon: "ic_theater_24", tint: "bg_icon_orange"),
        "health": Category(id: "health", title: "건강 (Health)", emoji: "❤️", icon: "ic_health_24", tint: "bg_icon_red"),
        "politic": Category(id: "politic", title: "정치 (Politics)", emoji: "💼", icon: "ic_account_balance_24", tint: "bg_icon_purple"),
        "science": Category(id: "science", title: "과학 (Science)", emoji: "🔬", icon: "ic_beaker_24", tint: "bg_icon_green"),
        "social": Category(id: "social", title: "사회 (Social)", emoji: "👥", icon: "ic_groups_24", tint: "bg_icon_orange"),
    ]

    /// 주제별 콘텐츠 주제 폴더 -> 한글 제목
    static let articleLessons: [String: String] = [
        "architect": "건축과 디자인 (Architecture & Design)",
        "art": "미술과 예술 (Art)",
        "city": "도시와 건축 (City & Architecture)",
        "festival": "축제와 행사 (Festival)",
        "instrument": "악기와 음악 (Musical Instruments)",
        "local_food": "지역 음식과 특산품 (Local Food & Specialties)",
        "museum": "박물관과 전시 (Museum)",
        "performance": "공연과 무대 (Performance)",
        "unesco": "유네스코 문화유산 (UNESCO)",
        "behavior": "행동경제학 (Behavioral Economics)",
        "bitcoin": "비트코인과 암호화폐 (Bitcoin)",
        "currency": "통화 (Currency)",
        "digital_economics": "디지털 경제 (Digital Economics)",
        "exchange_rate": "환율 (Exchange Rate)",
        "macro": "거시경제학 (Macroeconomics)",
        "micro": "미시경제학 (Microeconomics)",
        "stock": "주식 (Stock)",
        "supply_demand": "수요와 공급 (Supply & Demand)",
        "trend": "경제 트렌드 (Economic Trend)",
        "e-sports": "e-스포츠 (e-Sports)",
        "k-pop": "k-pop (K-Pop)",
        "k-webtoon": "k-웹툰 (K-Webtoon)",
        "movie_casting": "영화 캐스팅 (Movie Casting)",
        "movie_music": "영화 음악 (Movie Music)",
        "music_genere": "음악 장르 (Music Genre)",
        "musical": "뮤지컬 (Musical)",
        "ott": "OTT (Over-the-Top)",
        "soccer": "축구 (Soccer)",
        "swimming": "수영 (Swimming)",
        "addiction": "중독과 회복 (Addiction)",
        "disease": "질병과 치료 (Disease)",
        "hormon": "호르몬과 내분비 (Hormone)",
        "mental": "정신건강 (Mental Health)",
        "mucle": "근육과 운동 (Muscle)",
        "obesity": "비만과 관리 (Obesity)",
        "sleep": "수면과 휴식 (Sleep)",
        "stress": "스트레스 관리 (Stress)",
        "vitamin": "비타민과 영양소 (Vitamin)",
        "work_out": "운동과 피트니스 (Work Out)",
        "democracy": "민주주의 (Democracy)",
        "dictator": "독재 (Dictator)",
        "diplomacy": "외교 (Diplomacy)",
        "election": "선거 (Election)",
        "governments": "정부 (Government)",
        "law": "법률 (Law)",
        "parliamentary": "의회 (Parliamentary)",
        "party": "정당 (Party)",
        "security": "안전 (Security)",
        "social": "사회 (Social)",
        "battery": "배터리 기술 (Battery Technology)",
        "computer": "컴퓨터과학 (Computer Science)",
        "earth": "지구과학 (Earth Science)",
        "material": "소재과학 (Material Science)",
        "neuro_science": "신경과학 (Neuroscience)",
        "physics": "물리학 (Physics)",
        "robot": "로봇공학 (Robot)",
        "uam": "도시 항공 모빌리티 (UAM)",
        "universe": "우주 (Universe)",
        "aging_society": "고령화 사회 (aging society)",
        "crime": "범죄 (crime)",
        "elitism": "엘리트주의 (Elitism)",
        "gender": "성평등 (Gender Equality)",
        "generation_gap": "세대 갈등 (Generation Gap)",
        "house": "주택문제 (House)",
        "low_birth": "저출산 문제 (Lower Birth Rate)",
        "poverty": "빈곤과 불평등 (Poverty)",
        "racism": "인종차별 (Racism)",
        "welfare": "복지 정책 (Welfare)",
    ]

    /// 주제 폴더 -> 안드로이드 아이콘 이름 (주제별 콘텐츠용; 빈출표현은 Topic 에 들어 있다)
    static let lessonIcons: [String: String] = [
        "architect": "ic_building2_24",
        "art": "ic_palette_24",
        "city": "ic_building_24",
        "festival": "ic_party_popper_24",
        "instrument": "ic_music_24",
        "local_food": "ic_utensils_24",
        "museum": "ic_landmark_24",
        "performance": "ic_theater_24",
        "unesco": "ic_globe_24",
        "behavior": "ic_psychology_24",
        "bitcoin": "ic_bitcoin_24",
        "currency": "ic_dollar_24",
        "digital_economics": "ic_monitor_24",
        "exchange_rate": "ic_arrow_right_left_24",
        "macro": "ic_globe_chart_24",
        "micro": "ic_pie_chart_24",
        "stock": "ic_trending_up_24",
        "supply_demand": "ic_line_chart_24",
        "trend": "ic_trending_up_24",
        "e-sports": "ic_sports_esports_24",
        "k-pop": "ic_auto_awesome_24",
        "k-webtoon": "ic_book_open_24",
        "movie_casting": "ic_movie_24",
        "movie_music": "ic_queue_music_24",
        "music_genere": "ic_library_music_24",
        "musical": "ic_theaters_24",
        "ott": "ic_on_demand_video_24",
        "soccer": "ic_sports_soccer_24",
        "swimming": "ic_pool_24",
        "addiction": "ic_no_drinks_24",
        "disease": "ic_stethoscope_24",
        "hormon": "ic_flask_24",
        "mental": "ic_brain_24",
        "mucle": "ic_exercise_24",
        "obesity": "ic_monitor_weight_24",
        "sleep": "ic_bedtime_24",
        "stress": "ic_self_improvement_24",
        "vitamin": "ic_nutrition_24",
        "work_out": "ic_fitness_center_24",
        "democracy": "ic_how_to_vote_24",
        "dictator": "ic_no_accounts_24",
        "diplomacy": "ic_connecting_airports_24",
        "election": "ic_ballot_24",
        "governments": "ic_account_balance_24",
        "law": "ic_gavel_24",
        "parliamentary": "ic_assured_workload_24",
        "party": "ic_flag_24",
        "security": "ic_shield_24",
        "social": "ic_diversity_3_24",
        "battery": "ic_battery_24",
        "computer": "ic_computer_science_24",
        "earth": "ic_earth_science_24",
        "material": "ic_material_science_24",
        "neuro_science": "ic_brain_circuit_24",
        "physics": "ic_physics_24",
        "robot": "ic_bot_24",
        "uam": "ic_plane_24",
        "universe": "ic_earth_24",
        "aging_society": "ic_elderly_24",
        "crime": "ic_local_police_24",
        "elitism": "ic_crown_24",
        "gender": "ic_wc_24",
        "generation_gap": "ic_family_restroom_24",
        "house": "ic_house_24",
        "low_birth": "ic_child_friendly_24",
        "poverty": "ic_savings_24",
        "racism": "ic_userx_24",
        "welfare": "ic_heart_handshake_24",
    ]

    /// 5.speech — 화면에 나오는 순서
    static let speeches: [Speech] = [
        Speech(folder: "lincoln", title: "Abraham Lincoln's Speech", subtitle: "게티즈버그 연설 · 1863", image: ""),
        Speech(folder: "mandela", title: "Nelson Mandela's Speech", subtitle: "대통령 취임 연설 · 1994", image: ""),
        Speech(folder: "reagan", title: "Ronald Reagan's Speech", subtitle: "베를린 장벽 연설 · 발췌 · 1987", image: ""),
        Speech(folder: "conan", title: "Conan O'Brien's Speech", subtitle: "다트머스대 연설문", image: "conan"),
        Speech(folder: "barackobama", title: "Barack Obama's Speech", subtitle: "취임 연설문", image: "obama"),
        Speech(folder: "gergebush", title: "George Bush's Speech", subtitle: "911테러 추모 연설문", image: "bush"),
        Speech(folder: "jfkennedy", title: "John F. Kennedy's Speech", subtitle: "로버트프로스트 추모 연설문", image: "jfk"),
        Speech(folder: "martinLutherkingjr", title: "Martin Luther King Jr.'s Speech", subtitle: "I have a dream", image: "king"),
    ]

    /// 연설문 화면 아래에 붙는 안내 — 안드로이드 activity_level3 의 "연설문 학습 특징".
    static let speechHighlights: [(title: String, detail: String)] = [
        ("실제 명연설 학습", "세계적으로 유명한 연설문으로 고급 영어 표현을 익혀보세요"),
        ("감정과 뉘앙스 연습", "연설자의 감정과 의도를 담은 발음 연습"),
        ("고급 어휘와 문법", "격식있는 표현과 수사법을 자연스럽게 체득"),
    ]


}
