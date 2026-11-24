package com.chisnova.mercury.data

data class Lesson(
    val id: String,
    val title: String,
    val iconResId: Int,
    val isCompleted: Boolean = false,
    val sentences: List<String> = emptyList()
)

data class LessonCategory(
    val id: String,
    val title: String,
    val emoji: String,
    val iconResId: Int,
    val bgColorResId: Int,
    val lessons: List<Lesson>
)

object LessonData {
    
    // Level 1 데이터
    fun getLevel1Categories(): List<LessonCategory> {
        return listOf(
            LessonCategory(
                id = "basic",
                title = "기초",
                emoji = "📚",
                iconResId = com.chisnova.mercury.R.drawable.ic_book_open_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_blue,
                lessons = listOf(
                    Lesson(
                        id = "greetings",
                        title = "인사말과 기본 표현",
                        iconResId = com.chisnova.mercury.R.drawable.ic_users_24,
                        isCompleted = true,
                        sentences = listOf(
                            "Hello, nice to meet you!",
                            "How are you doing today?",
                            "Thank you very much.",
                            "Have a great day!",
                            "See you later!"
                        )
                    ),
                    Lesson(
                        id = "daily_conversation",
                        title = "일상 대화 표현",
                        iconResId = com.chisnova.mercury.R.drawable.ic_people_chat_24,
                        isCompleted = true,
                        sentences = listOf(
                            "What's the weather like today?",
                            "I'm going to the store.",
                            "Would you like some coffee?",
                            "That sounds like a good idea.",
                            "I'll see you tomorrow."
                        )
                    ),
                    Lesson(
                        id = "emotions",
                        title = "감정 표현하기",
                        iconResId = com.chisnova.mercury.R.drawable.ic_star_24,
                        isCompleted = true,
                        sentences = listOf(
                            "I'm really happy about this!",
                            "That makes me feel sad.",
                            "I'm excited for the weekend.",
                            "Don't worry, everything will be fine.",
                            "I'm proud of your achievement."
                        )
                    ),
                    Lesson(
                        id = "time_date",
                        title = "시간과 날짜 표현",
                        iconResId = com.chisnova.mercury.R.drawable.ic_clock_24,
                        isCompleted = false,
                        sentences = listOf(
                            "What time is it now?",
                            "It's three thirty in the afternoon.",
                            "I have a meeting at nine AM.",
                            "Today is Monday, March 15th.",
                            "I'll call you next week."
                        )
                    )
                )
            ),
            LessonCategory(
                id = "real_life",
                title = "실생활",
                emoji = "🏪",
                iconResId = com.chisnova.mercury.R.drawable.ic_coffee_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_green,
                lessons = listOf(
                    Lesson(
                        id = "food_ordering",
                        title = "음식 주문하기",
                        iconResId = com.chisnova.mercury.R.drawable.ic_coffee_24,
                        isCompleted = false,
                        sentences = listOf(
                            "I'd like to order a coffee, please.",
                            "Can I have the menu?",
                            "I'll take the chicken sandwich.",
                            "Could I get that to go?",
                            "The bill, please."
                        )
                    ),
                    Lesson(
                        id = "directions",
                        title = "방향 안내 (Directions)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_map_pin_24,
                        isCompleted = false,
                        sentences = listOf(
                            "Excuse me, where is the nearest subway station?",
                            "Go straight and turn left at the corner.",
                            "It's about a ten-minute walk from here.",
                            "You can't miss it, it's a big building.",
                            "Thank you for your help!"
                        )
                    ),
                    Lesson(
                        id = "asking_help",
                        title = "도움 요청하기(Request Help)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_help_circle_24,
                        isCompleted = false,
                        sentences = listOf(
                            "Could you help me with this?",
                            "I'm having trouble understanding.",
                            "Would you mind explaining that again?",
                            "I need some assistance, please.",
                            "That's very kind of you."
                        )
                    ),
                    Lesson(
                        id = "shopping",
                        title = "쇼핑하기 (Shopping & Ordering)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_shopping_cart_24,
                        isCompleted = false,
                        sentences = listOf(
                            "How much does this cost?",
                            "Do you have this in a different size?",
                            "I'm just looking, thank you.",
                            "Can I pay by credit card?",
                            "I'd like to return this item."
                        )
                    )
                )
            ),
            LessonCategory(
                id = "communication",
                title = "소통",
                emoji = "💬",
                iconResId = com.chisnova.mercury.R.drawable.ic_message_circle_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_blue,
                lessons = listOf(
                    Lesson(
                        id = "apologies",
                        title = "사과하기 (Apologies)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_volunteer_activism_24,
                        isCompleted = false,
                        sentences = listOf(
                            "I'm sorry for being late.",
                            "I apologize for the confusion.",
                            "That was my mistake.",
                            "I didn't mean to upset you.",
                            "Please forgive me."
                        )
                    ),
                    Lesson(
                        id = "making_plans",
                        title = "계획 세우기(Making Plans)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_edit_calendar_24,
                        isCompleted = false,
                        sentences = listOf(
                            "What are you doing this weekend?",
                            "Would you like to have dinner together?",
                            "Let's meet at seven o'clock.",
                            "How about we go to the movies?",
                            "That sounds like a great plan!"
                        )
                    ),
                    Lesson(
                        id = "opinions_feelings",
                        title = "의견 표현 (Opinions & Feelings)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_thumbs_up_down_24,
                        isCompleted = false,
                        sentences = listOf(
                            "In my opinion, this is the best choice.",
                            "I think you're absolutely right.",
                            "I'm not sure I agree with that.",
                            "How do you feel about this?",
                            "I have mixed feelings about it."
                        )
                    ),
                    Lesson(
                        id = "polite_disagreement",
                        title = "예의있는 거절(Polite Disagreement)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_x_24,
                        isCompleted = false,
                        sentences = listOf(
                            "I'm afraid I can't agree with that.",
                            "I see your point, but I think differently.",
                            "I'm sorry, but I have to decline.",
                            "That's not quite how I see it.",
                            "I appreciate the offer, but no thank you."
                        )
                    ),
                    Lesson(
                        id = "small_talk",
                        title = "스몰토크(Small Talk)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_wb_cloudy_24,
                        isCompleted = false,
                        sentences = listOf(
                            "Beautiful weather we're having, isn't it?",
                            "How was your weekend?",
                            "Did you watch the game last night?",
                            "Have you been to that new restaurant?",
                            "Any plans for the holidays?"
                        )
                    ),
                    Lesson(
                        id = "suggestions_advice",
                        title = "제안하기(Suggestion & Advice)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_lightbulb_24,
                        isCompleted = false,
                        sentences = listOf(
                            "Why don't you try this approach?",
                            "I suggest you talk to your manager.",
                            "You might want to consider this option.",
                            "If I were you, I would do it differently.",
                            "Have you thought about taking a break?"
                        )
                    )
                )
            ),
            LessonCategory(
                id = "important",
                title = "중요",
                emoji = "🚨",
                iconResId = com.chisnova.mercury.R.drawable.ic_alert_circle_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_red,
                lessons = listOf(
                    Lesson(
                        id = "emergency",
                        title = "비상상황 대처 (Emergency)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_mic_24,
                        isCompleted = false,
                        sentences = listOf(
                            "Help! I need assistance!",
                            "Please call an ambulance.",
                            "There's been an accident.",
                            "I need to speak to the police.",
                            "This is an emergency situation."
                        )
                    )
                )
            )
        )
    }
    
    // Level 2 데이터
    fun getLevel2Categories(): List<LessonCategory> {
        return listOf(
            LessonCategory(
            id = "culture",
            title = "문화 (Culture)",
            emoji = "🎨",
            iconResId = com.chisnova.mercury.R.drawable.ic_palette_24,
            bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_green,
            lessons = listOf(
                    Lesson(
                        id = "architecture",
                        title = "건축과 디자인 (Architecture & Design)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_building2_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "art",
                        title = "미술과 예술 (Art)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_palette_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "city",
                        title = "도시와 건축 (City & Architecture)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_building_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "festival",
                        title = "축제와 행사 (Festival)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_party_popper_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "instrument",
                        title = "악기와 음악 (Musical Instruments)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_music_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "local_food",
                        title = "지역 음식과 특산품 (Local Food & Specialties)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_utensils_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "museum",
                        title = "박물관과 전시 (Museum)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_landmark_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "performance",
                        title = "공연과 무대 (Performance)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_theater_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "unesco",
                        title = "유네스코 문화유산 (UNESCO)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_globe_24,
                        isCompleted = false,
                        sentences = listOf()
                    )
                )
            ),

            LessonCategory(
                id = "economics",
                title = "경제 (Economics)",
                emoji = "💰",
                iconResId = com.chisnova.mercury.R.drawable.ic_dollar_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_blue,
                lessons = listOf(
                    Lesson(
                        id = "behavior",
                        title = "행동경제학 (Behavioral Economics)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_psychology_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "bitcoin",
                        title = "비트코인과 암호화폐 (Bitcoin)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_bitcoin_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "currency",
                        title = "통화 (Currency)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_dollar_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "digital_economics",
                        title = "디지털 경제 (Digital Economics)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_monitor_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "exchange_rate",
                        title = "환율 (Exchange Rate)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_arrow_right_left_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "macro",
                        title = "거시경제학 (Macroeconomics)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_globe_chart_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "micro",
                        title = "미시경제학 (Microeconomics)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_pie_chart_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "stock",
                        title = "주식 (Stock)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_trending_up_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "supply_demand",
                        title = "수요와 공급 (Supply & Demand)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_line_chart_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "trend",
                        title = "경제 트렌드 (Economic Trend)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_trending_up_24,
                        isCompleted = false,
                        sentences = listOf()
                    )
                )
            ),
            LessonCategory(
                id = "entertain",
                title = "엔터테인먼트 (Entertainment)",
                emoji = "🎬",
                iconResId = com.chisnova.mercury.R.drawable.ic_theater_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_orange,
                lessons = listOf(
                    Lesson(
                        id = "e-sports",
                        title = "e-스포츠 (e-Sports)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_sports_esports_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "k-pop",
                        title = "k-pop (K-Pop)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_auto_awesome_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "k-webtoon",
                        title = "k-웹툰 (K-Webtoon)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_book_open_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "movie_casting",
                        title = "영화 캐스팅 (Movie Casting)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_movie_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "movie_music",
                        title = "영화 음악 (Movie Music)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_queue_music_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "music_genre",
                        title = "음악 장르 (Music Genre)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_library_music_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "musical",
                        title = "뮤지컬 (Musical)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_theaters_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "ott",
                        title = "OTT (Over-the-Top)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_on_demand_video_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "soccer",
                        title = "축구 (Soccer)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_sports_soccer_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "swimming",
                        title = "수영 (Swimming)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_pool_24,
                        isCompleted = false,
                        sentences = listOf()
                    )
                )
            ),
            LessonCategory(
                id = "health",
                title = "건강 (Health)",
                emoji = "❤️",
                iconResId = com.chisnova.mercury.R.drawable.ic_health_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_red,
                lessons = listOf(
                    Lesson(
                        id = "addiction",
                        title = "중독과 회복 (Addiction)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_no_drinks_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "disease",
                        title = "질병과 치료 (Disease)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_stethoscope_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "hormon",
                        title = "호르몬과 내분비 (Hormone)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_flask_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "mental",
                        title = "정신건강 (Mental Health)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_brain_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "muscle",
                        title = "근육과 운동 (Muscle)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_exercise_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "obesity",
                        title = "비만과 관리 (Obesity)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_monitor_weight_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "sleep",
                        title = "수면과 휴식 (Sleep)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_bedtime_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "stress",
                        title = "스트레스 관리 (Stress)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_self_improvement_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "vitamin",
                        title = "비타민과 영양소 (Vitamin)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_nutrition_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "work_out",
                        title = "운동과 피트니스 (Work Out)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_fitness_center_24,
                        isCompleted = false,
                        sentences = listOf()
                    )
                )
            ),
            LessonCategory(
                id = "politic",
                title = "정치 (Politics)",
                emoji = "💼",
                iconResId = com.chisnova.mercury.R.drawable.ic_account_balance_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_purple,
                lessons = listOf(
                    Lesson(
                        id = "democracy",
                        title = "민주주의 (Democracy)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_how_to_vote_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "dictator",
                        title = "독재 (Dictator)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_no_accounts_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "diplomacy",
                        title = "외교 (Diplomacy)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_connecting_airports_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "election",
                        title = "선거 (Election)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_ballot_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "government",
                        title = "정부 (Government)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_account_balance_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "law",
                        title = "법률 (Law)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_gavel_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "parliamentary",
                        title = "의회 (Parliamentary)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_assured_workload_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "party",
                        title = "정당 (Party)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_flag_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "security",
                        title = "안전 (Security)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_shield_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "social",
                        title = "사회 (Social)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_diversity_3_24,
                        isCompleted = false,
                        sentences = listOf()
                    )
                )
            ),
            LessonCategory(
                id = "science",
                title = "과학 (Science)",
                emoji = "🔬",
                iconResId = com.chisnova.mercury.R.drawable.ic_beaker_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_green,
                lessons = listOf(
                    Lesson(
                        id = "battery",
                        title = "배터리 기술 (Battery Technology)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_battery_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "computer",
                        title = "컴퓨터과학 (Computer Science)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_computer_science_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "earth",
                        title = "지구과학 (Earth Science)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_earth_science_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "material",
                        title = "소재과학 (Material Science)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_material_science_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "neuro_science",
                        title = "신경과학 (Neuroscience)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_brain_circuit_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "physics",
                        title = "물리학 (Physics)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_physics_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "robot",
                        title = "로봇공학 (Robot)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_bot_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "uam",
                        title = "도시 항공 모빌리티 (UAM)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_plane_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "universe",
                        title = "우주 (Universe)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_earth_24,
                        isCompleted = false,
                        sentences = listOf()
                    )
                )
            ),
            LessonCategory(
                id = "social",
                title = "사회 (Social)",
                emoji = "👥",
                iconResId = com.chisnova.mercury.R.drawable.ic_groups_24,
                bgColorResId = com.chisnova.mercury.R.drawable.bg_icon_orange,
                lessons = listOf(
                    Lesson(
                        id = "aging_society",
                        title = "고령화 사회 (aging society)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_elderly_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "crime",
                        title = "범죄 (crime)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_local_police_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "elitism",
                        title = "엘리트주의 (Elitism)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_crown_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "gender_equality",
                        title = "성평등 (Gender Equality)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_wc_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "generation_gap",
                        title = "세대 갈등 (Generation Gap)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_family_restroom_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "house",
                        title = "주택문제 (House)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_house_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "lower_birth",
                        title = "저출산 문제 (Lower Birth Rate)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_child_friendly_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "poverty",
                        title = "빈곤과 불평등 (Poverty)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_savings_24,
                        isCompleted = true,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "racism",
                        title = "인종차별 (Racism)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_userx_24,
                        isCompleted = false,
                        sentences = listOf()
                    ),
                    Lesson(
                        id = "welfare",
                        title = "복지 정책 (Welfare)",
                        iconResId = com.chisnova.mercury.R.drawable.ic_heart_handshake_24,
                        isCompleted = false,
                        sentences = listOf()
                    )
                )
            )
        )
    }
}