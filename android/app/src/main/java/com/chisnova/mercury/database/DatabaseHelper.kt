package com.chisnova.mercury.database

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.content.ContentValues
import android.database.Cursor
import java.io.File
import java.io.FileOutputStream
import android.util.Log

class DatabaseHelper(private val appContext: Context) : SQLiteOpenHelper(appContext, DATABASE_NAME, null, DATABASE_VERSION) {
    init {
        installDatabaseIfNeeded(appContext)
    }
 
    companion object {
        private const val DATABASE_NAME = "mercury.db"
        private const val DATABASE_VERSION = 4
        
        @Volatile
        private var INSTANCE: DatabaseHelper? = null
        
        fun getInstance(context: Context): DatabaseHelper {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: DatabaseHelper(context.applicationContext).also { INSTANCE = it }
            }
        }
        
        // Singleton database connection
        @Volatile
        private var database: SQLiteDatabase? = null
        
        private fun getDatabase(context: Context): SQLiteDatabase {
			return database ?: synchronized(this) {
				database ?: getInstance(context).readableDatabase.also { database = it }
			}
        }

    }

    override fun onCreate(db: SQLiteDatabase) {
        // Database is created from assets, so no need to create tables here
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            // 제외한 연설의 본문만 제거하고 사용자의 연습 기록은 보존한다.
            db.delete("Contents", "CategoryName = ?", arrayOf("jkrolling"))
        }
        if (oldVersion < 3) {
            // 본문만 제외하고 기존 연습 기록은 보존한다.
            db.delete("Contents", "CategoryName = ?", arrayOf("stevejobs"))
        }
        if (oldVersion < 4) installNewSpeeches(db)
    }

    // 기존 연습 기록과 충돌하지 않도록 제작 시 부여한 고정 UID를 사용한다.
    private fun installNewSpeeches(db: SQLiteDatabase) {
        for (name in listOf("lincoln", "mandela", "reagan")) {
            val base = "contents/5.speech/$name/$name"
            fun readText(extension: String) = appContext.assets.open("$base.$extension")
                .bufferedReader().use { it.readText() }
            val metadata = org.json.JSONObject(readText("metadata.json"))
            val english = readText("eng").lineSequence().filter { it.isNotBlank() }.toList()
            val korean = readText("kor").lineSequence().filter { it.isNotBlank() }.toList()
            val durations = metadata.getJSONArray("audioDurationsMs")
            require(english.size == korean.size && english.size == durations.length())
            for (index in english.indices) {
                val uid = metadata.getInt("uidStart") + index
                db.rawQuery("SELECT CategoryName, SentenceID FROM Contents WHERE UID=?",
                    arrayOf(uid.toString())).use { cursor ->
                    if (cursor.moveToFirst()) {
                        check(cursor.getString(0) == name && cursor.getInt(1) == index)
                    }
                }
                val values = ContentValues().apply {
                    put("UID", uid)
                    put("CategoryType", 2)
                    put("ContentsID", metadata.getInt("contentsId"))
                    put("SentenceID", index)
                    put("CategoryName", name)
                    put("EngLabel", english[index])
                    put("KorLabel", korean[index])
                    put("LabelAudioSize", durations.getInt(index))
                }
                if (db.update("Contents", values, "UID=?", arrayOf(uid.toString())) == 0) {
                    db.insertOrThrow("Contents", null, values)
                }
            }
        }
    }

    private fun installDatabaseIfNeeded(context: Context) {
        val dbFile = context.getDatabasePath(DATABASE_NAME)
        if (dbFile.exists()) return
        dbFile.parentFile?.let { parent ->
            if (!parent.exists()) parent.mkdirs()
        }
        try {
            context.assets.open(DATABASE_NAME).use { input ->
                FileOutputStream(dbFile).use { output ->
                    val buffer = ByteArray(8 * 1024)
                    var read = input.read(buffer)
                    while (read != -1) {
                        output.write(buffer, 0, read)
                        read = input.read(buffer)
                    }
                    output.flush()
                }
            }
        } catch (e: Exception) {
            throw RuntimeException("Failed to install database from assets: $DATABASE_NAME", e)
        }
    }

    // Query BasicContents table
    fun queryContents(categoryName: String? = null): List<ContentItem> {
        Log.i("[AsrAPI]", "queryContents categoryName: $categoryName")
        val db = readableDatabase
        Log.i("[AsrAPI]", "queryContents db: $db")
        
        val cursor = db.query(
            "Contents",
            arrayOf("UID", "CategoryType", "ContentsID", "SentenceID", "CategoryName", "EngLabel", "KorLabel", "LabelAudioSize"),
            "CategoryName = ?",
            arrayOf(categoryName),
            null,
            null,
            "SentenceID ASC"
        )
        Log.i("[AsrAPI]", "queryContents cursor: $cursor")
        return cursorToContentItems(cursor)
    }

    fun queryAttemptsByUid(uid: Int): List<AttemptItem> {
        val db = readableDatabase
        val items = mutableListOf<AttemptItem>()
        
        val cursor = db.query(
            "Attempt",
            null,
            "UID = ?",
            arrayOf(uid.toString()),
            null,
            null,
            "Date DESC"
        )
        
        cursor?.use {
            while (it.moveToNext()) {
                val date = it.getString(it.getColumnIndexOrThrow("Date"))
                val aid = it.getString(it.getColumnIndexOrThrow("AID"))
                val attemptUid = it.getInt(it.getColumnIndexOrThrow("UID"))
                val numWord = it.getInt(it.getColumnIndexOrThrow("NumWord"))
                val transcript = it.getString(it.getColumnIndexOrThrow("Transcript"))
                val recordPath = it.getString(it.getColumnIndexOrThrow("RecordPath"))
                val fluency = it.getFloat(it.getColumnIndexOrThrow("Fluency"))
                val pronounce = it.getFloat(it.getColumnIndexOrThrow("Pronounce"))
                
                items.add(
                    AttemptItem(
                        date = date,
                        aid = aid,
                        uid = attemptUid,
                        numWord = numWord,
                        transcript = transcript,
                        recordPath = recordPath,
                        fluency = fluency,
                        pronounce = pronounce
                    )
                )
            }
        }
        return items
    }
    
    /**
     * 특정 UID의 가장 최근 Attempt 한 건만 반환한다.
     * 기존 queryAttemptsByUid()가 전체 목록을 가져오고 [0]만 쓰는 N+1 패턴을 개선하기 위해 추가.
     * LIMIT 1로 DB에서 직접 한 건만 읽으므로 다량의 문장 로딩 시 불필요한 커서 순회를 없앤다.
     */
    fun queryLatestAttemptByUid(uid: Int): AttemptItem? {
        val db = readableDatabase
        val cursor = db.query(
            "Attempt",
            null,
            "UID = ?",
            arrayOf(uid.toString()),
            null,
            null,
            "Date DESC",
            "1"   // LIMIT 1
        )
        return cursor?.use {
            if (it.moveToFirst()) {
                AttemptItem(
                    date = it.getString(it.getColumnIndexOrThrow("Date")),
                    aid = it.getString(it.getColumnIndexOrThrow("AID")),
                    uid = it.getInt(it.getColumnIndexOrThrow("UID")),
                    numWord = it.getInt(it.getColumnIndexOrThrow("NumWord")),
                    transcript = it.getString(it.getColumnIndexOrThrow("Transcript")),
                    recordPath = it.getString(it.getColumnIndexOrThrow("RecordPath")),
                    fluency = it.getFloat(it.getColumnIndexOrThrow("Fluency")),
                    pronounce = it.getFloat(it.getColumnIndexOrThrow("Pronounce"))
                )
            } else null
        }
    }

    private fun cursorToContentItems(cursor: Cursor): List<ContentItem> {
        val items = mutableListOf<ContentItem>()
        
        cursor.use {
            while (it.moveToNext()) {
                val uid = it.getInt(it.getColumnIndexOrThrow("UID"))
                val categoryType = it.getInt(it.getColumnIndexOrThrow("CategoryType"))
                val contentsId = it.getInt(it.getColumnIndexOrThrow("ContentsID"))
                val sentenceId = it.getInt(it.getColumnIndexOrThrow("SentenceID"))
                val categoryName = it.getString(it.getColumnIndexOrThrow("CategoryName"))
                val engLabel = it.getString(it.getColumnIndexOrThrow("EngLabel"))
                val korLabel = it.getString(it.getColumnIndexOrThrow("KorLabel"))
                val labelAudioSize = it.getInt(it.getColumnIndexOrThrow("LabelAudioSize"))
                
                items.add(
                    ContentItem(
                        uid = uid,
                        categoryType = categoryType,
                        contentsId = contentsId,
                        sentenceId = sentenceId,
                        categoryName = categoryName,
                        engLabel = engLabel,
                        korLabel = korLabel,
                        labelAudioSize = labelAudioSize
                    )
                )
            }
        }
        
        return items
    }

    fun insertAttempt(
        date: String,
        aid: String,
        uid: Int,
        numWord: Int,
        transcript: String,
        recordPath: String,
        fluency: Float,
        pronounce: Float,

    ): Long {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("Date", date)
            put("AID", aid)
            put("UID", uid)
            put("NumWord", numWord)
            put("Transcript", transcript)
            put("RecordPath", recordPath)
            put("Fluency", fluency)
            put("Pronounce", pronounce)
        }
        
        return db.insert("Attempt", null, values)
    }

    fun insertWordHistory(
        date: String,
        aid: String,
        wordIdx: Int,
        word: String,
        numPhone: Int,
        color: String,
        score: Float,
        comment: String
    ): Long {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("Date", date)
            put("AID", aid)
            put("WordIdx", wordIdx)
            put("Word", word)
            put("NumPhone", numPhone)
            put("Color", color)
            put("Score", score)
            put("Comment", comment)
        }
        
        return db.insert("WordHistory", null, values)
    }

    fun queryWordHistoryWithAID(
        aid: String
    ): List<WordHistoryItem> {
        val db = readableDatabase
        val cursor = db.query(
            "WordHistory",
            arrayOf("Date", "AID", "WordIdx", "Word", "NumPhone", "Color", "Score", "Comment"),
            "AID = ?",
            arrayOf(aid),
            null,
            null,
            null
        )
        return cursorToWordHistoryItems(cursor)
    }

    fun insertPhoneHistory(
        date: String,
        aid: String,
        wordIdx: Int,
        phoneIdx: Int,
        phone: String,
        category: String,
        score: Float
    ): Long {
        val db = writableDatabase
        val values = ContentValues().apply {
            put("Date", date)
            put("AID", aid)
            put("WordIdx", wordIdx)
            put("PhoneIdx", phoneIdx)
            put("Phone", phone)
            put("Category", category)
            put("Score", score)
        }
        
        return db.insert("PhoneHistory", null, values)
    }

    fun queryPhoneHistoryWithAID(
        aid: String
        ): List<PhoneHistoryItem> {
            val db = readableDatabase
            val cursor = db.query(
                "PhoneHistory",
                arrayOf("Date", "AID", "WordIdx", "PhoneIdx", "Phone", "Category", "Score"),
                "AID = ?",
                arrayOf(aid),
                null,
                null,
                null
            )
            return cursorToPhoneHistoryItems(cursor)
        }

    fun queryCategoryScores(startDate: String? = null): Map<String, Float> {
        val db = readableDatabase
        val query = StringBuilder("SELECT Category, AVG(Score) FROM PhoneHistory")
        val args = mutableListOf<String>()

        if (startDate != null) {
            query.append(" WHERE Date >= ?")
            args.add(startDate)
        }
        query.append(" GROUP BY Category")

        val cursor = db.rawQuery(query.toString(), args.toTypedArray())
        val result = mutableMapOf<String, Float>()
        
        cursor.use {
            while (it.moveToNext()) {
                val category = it.getString(0)
                val score = it.getFloat(1)
                if (!category.isNullOrBlank()) {
                     result[category] = score
                }
            }
        }
        return result
    }

    fun getAverageAttemptScore(): Int {
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT AVG(Pronounce) FROM Attempt", null)
        var avg = 0f
        if (cursor.moveToFirst()) {
            avg = cursor.getFloat(0)
        }
        cursor.close()
        return avg.toInt()
    }

    fun getTotalAttemptsCount(): Int {
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT COUNT(*) FROM Attempt", null)
        var count = 0
        if (cursor.moveToFirst()) {
            count = cursor.getInt(0)
        }
        cursor.close()
        return count
    }

    fun getTotalLearnedWordsCount(): Int {
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT COUNT(*) FROM WordHistory", null)
        var count = 0
        if (cursor.moveToFirst()) {
            count = cursor.getInt(0)
        }
        cursor.close()
        return count
    }

    fun getImprovementNeededCount(): Int {
        val db = readableDatabase
        val cursor = db.rawQuery("SELECT COUNT(*) FROM WordHistory WHERE Score < 60", null)
        var count = 0
        if (cursor.moveToFirst()) {
            count = cursor.getInt(0)
        }
        cursor.close()
        return count
    }

    fun getWeakestPhones(startDate: String? = null): List<WeakPhoneItem> {
        val db = readableDatabase
        val query = StringBuilder("SELECT Phone, Category, AVG(Score) as AvgScore FROM PhoneHistory")
        val args = mutableListOf<String>()

        if (startDate != null) {
            query.append(" WHERE Date >= ?")
            args.add(startDate)
        }
        
        query.append(" GROUP BY Phone ORDER BY AvgScore ASC LIMIT 3")

        val cursor = db.rawQuery(query.toString(), args.toTypedArray())
        val items = mutableListOf<WeakPhoneItem>()

        cursor.use {
            while (it.moveToNext()) {
                val phone = it.getString(0)
                val category = it.getString(1)
                val score = it.getFloat(2).toInt()
                items.add(WeakPhoneItem(phone, category, score))
            }
        }
        return items
    }

    /**
     * 복수 UID에 대해 각각 최신 Attempt 1건씩을 한 번의 쿼리로 반환한다.
     *
     * 기존 방식: 304개 UID × queryLatestAttemptByUid() = 304회 쿼리
     * 개선 방식:
     *   SELECT * FROM Attempt
     *   WHERE AID IN (
     *     SELECT AID FROM Attempt WHERE UID IN (...) GROUP BY UID HAVING AID = MAX(AID)
     *   )
     * 단일 쿼리로 304건의 최신 Attempt를 일괄 조회.
     *
     * 반환: Map<uid, AttemptItem>
     */
    /**
     * 홈 화면 통계용 — 모든 시도의 (날짜, uid) 만 가볍게 가져온다.
     * 진도·연속일·주간 그래프를 그리는 데는 이 두 값이면 충분하다.
     */
    fun queryAllAttemptDates(visibleOnly: Boolean = false): List<Pair<String, Int>> {
        val out = mutableListOf<Pair<String, Int>>()
        val hidden = if (visibleOnly) hiddenProgressAttemptIds() else emptySet()
        readableDatabase.rawQuery("SELECT Date, UID, AID FROM Attempt ORDER BY Date DESC", null).use {
            while (it.moveToNext()) {
                if (it.getString(2) !in hidden) out.add(it.getString(0) to it.getInt(1))
            }
        }
        return out
    }

    private fun hiddenProgressAttemptIds(): Set<String> = appContext
        .getSharedPreferences("learning_progress", Context.MODE_PRIVATE)
        .getStringSet("hidden_attempt_ids", emptySet())?.toSet() ?: emptySet()

    /** 기록을 삭제하지 않고 화면과 진행률에서만 숨기거나 복원한다. 새 시도는 항상 표시된다. */
    fun setProgressHidden(category: String, hidden: Boolean) {
        val topicIds = mutableSetOf<String>()
        readableDatabase.rawQuery(
            "SELECT a.AID FROM Attempt a JOIN Contents c ON a.UID = c.UID WHERE c.CategoryName = ?",
            arrayOf(category)
        ).use { cursor ->
            while (cursor.moveToNext()) topicIds.add(cursor.getString(0))
        }
        val updated = hiddenProgressAttemptIds().toMutableSet()
        if (hidden) updated.addAll(topicIds) else updated.removeAll(topicIds)
        val saved = appContext.getSharedPreferences("learning_progress", Context.MODE_PRIVATE)
            .edit().putStringSet("hidden_attempt_ids", updated).commit()
        check(saved) { "학습 상태를 저장하지 못했습니다." }
    }

    /** uid -> 주제 폴더 이름. 시도가 어느 레벨인지 가리는 데 쓴다. */
    fun queryUidCategories(): Map<Int, String> {
        val out = mutableMapOf<Int, String>()
        readableDatabase.rawQuery("SELECT UID, CategoryName FROM Contents", null).use {
            while (it.moveToNext()) {
                out[it.getInt(0)] = it.getString(1)
            }
        }
        return out
    }

    fun queryLatestAttemptsForUids(uids: List<Int>, visibleOnly: Boolean = false): Map<Int, AttemptItem> {
        if (uids.isEmpty()) return emptyMap()
        val db = readableDatabase

        // SQLite에서 각 UID의 최신 Attempt를 한 번에 가져오기 위해
        // MAX(Date)로 그룹화 후 실제 행을 JOIN해서 반환.
        // IN 절 placeholders 생성
        val placeholders = uids.joinToString(",") { "?" }
        val args = uids.map { it.toString() }.toTypedArray()

        // 서브쿼리: UID별 MAX(Date) → 해당 행의 AID를 특정
        val sql = """
            SELECT a.Date, a.AID, a.UID, a.NumWord, a.Transcript, a.RecordPath, a.Fluency, a.Pronounce
            FROM Attempt a
            INNER JOIN (
                SELECT UID, MAX(Date) AS MaxDate
                FROM Attempt
                WHERE UID IN ($placeholders)
                GROUP BY UID
            ) AS latest ON a.UID = latest.UID AND a.Date = latest.MaxDate
            ORDER BY a.Date DESC, a.rowid DESC
        """.trimIndent()

        val cursor = db.rawQuery(sql, args)
        val result = mutableMapOf<Int, AttemptItem>()
        val hidden = if (visibleOnly) hiddenProgressAttemptIds() else emptySet()
        cursor.use {
            while (it.moveToNext()) {
                if (it.getString(1) in hidden) continue
                val item = AttemptItem(
                    date = it.getString(0),
                    aid = it.getString(1),
                    uid = it.getInt(2),
                    numWord = it.getInt(3),
                    transcript = it.getString(4),
                    recordPath = it.getString(5),
                    fluency = it.getFloat(6),
                    pronounce = it.getFloat(7)
                )
                // 같은 초에 여러 번 연습했어도 마지막으로 저장한 기록을 선택한다.
                if (!result.containsKey(item.uid)) {
                    result[item.uid] = item
                }
            }
        }
        return result
    }

    /**
     * 복수 AID에 대해 WordHistory를 한 번의 쿼리로 반환한다.
     * 반환: Map<aid, List<WordHistoryItem>>
     */
    fun queryWordHistoriesForAids(aids: List<String>): Map<String, List<WordHistoryItem>> {
        if (aids.isEmpty()) return emptyMap()
        val db = readableDatabase
        val placeholders = aids.joinToString(",") { "?" }
        val cursor = db.query(
            "WordHistory",
            arrayOf("Date", "AID", "WordIdx", "Word", "NumPhone", "Color", "Score", "Comment"),
            "AID IN ($placeholders)",
            aids.toTypedArray(),
            null, null, null
        )
        val result = mutableMapOf<String, MutableList<WordHistoryItem>>()
        cursor.use {
            while (it.moveToNext()) {
                val aid = it.getString(it.getColumnIndexOrThrow("AID"))
                val item = WordHistoryItem(
                    date = it.getString(it.getColumnIndexOrThrow("Date")),
                    aid = aid,
                    wordIdx = it.getInt(it.getColumnIndexOrThrow("WordIdx")),
                    word = it.getString(it.getColumnIndexOrThrow("Word")),
                    numPhone = it.getInt(it.getColumnIndexOrThrow("NumPhone")),
                    color = it.getString(it.getColumnIndexOrThrow("Color")),
                    score = it.getFloat(it.getColumnIndexOrThrow("Score")),
                    comment = it.getString(it.getColumnIndexOrThrow("Comment"))
                )
                result.getOrPut(aid) { mutableListOf() }.add(item)
            }
        }
        return result
    }

    /**
     * 복수 AID에 대해 PhoneHistory를 한 번의 쿼리로 반환한다.
     * 반환: Map<aid, List<PhoneHistoryItem>>
     */
    fun queryPhoneHistoriesForAids(aids: List<String>): Map<String, List<PhoneHistoryItem>> {
        if (aids.isEmpty()) return emptyMap()
        val db = readableDatabase
        val placeholders = aids.joinToString(",") { "?" }
        val cursor = db.query(
            "PhoneHistory",
            arrayOf("Date", "AID", "WordIdx", "PhoneIdx", "Phone", "Category", "Score"),
            "AID IN ($placeholders)",
            aids.toTypedArray(),
            null, null, null
        )
        val result = mutableMapOf<String, MutableList<PhoneHistoryItem>>()
        cursor.use {
            while (it.moveToNext()) {
                val aid = it.getString(it.getColumnIndexOrThrow("AID"))
                val item = PhoneHistoryItem(
                    date = it.getString(it.getColumnIndexOrThrow("Date")),
                    aid = aid,
                    wordIdx = it.getInt(it.getColumnIndexOrThrow("WordIdx")),
                    phoneIdx = it.getInt(it.getColumnIndexOrThrow("PhoneIdx")),
                    phone = it.getString(it.getColumnIndexOrThrow("Phone")),
                    category = it.getString(it.getColumnIndexOrThrow("Category")),
                    score = it.getFloat(it.getColumnIndexOrThrow("Score"))
                )
                result.getOrPut(aid) { mutableListOf() }.add(item)
            }
        }
        return result
    }

    fun deleteHistoryForUid(uid: Int) {
        val db = writableDatabase
        // First find AIDs
        val cursor = db.query("Attempt", arrayOf("AID"), "UID = ?", arrayOf(uid.toString()), null, null, null)
        val aids = mutableListOf<String>()
        cursor.use {
            while (it.moveToNext()) {
                aids.add(it.getString(0))
            }
        }
        
        db.beginTransaction()
        try {
            for (aid in aids) {
                db.delete("WordHistory", "AID = ?", arrayOf(aid))
                db.delete("PhoneHistory", "AID = ?", arrayOf(aid))
            }
            db.delete("Attempt", "UID = ?", arrayOf(uid.toString()))
            db.setTransactionSuccessful()
        } catch (e: Exception) {
            Log.e("DatabaseHelper", "Error deleting history for uid $uid", e)
        } finally {
            db.endTransaction()
        }
    }

    fun getHistoryUiItems(filter: String = "all"): List<HistoryUiItem> {
        val db = readableDatabase
        val items = mutableListOf<HistoryUiItem>()
        
        var query = """
            SELECT a.AID, a.Date, a.Pronounce, c.CategoryName, c.EngLabel, c.CategoryType 
            FROM Attempt a 
            LEFT JOIN Contents c ON a.UID = c.UID
        """
        
        val args = mutableListOf<String>()
        var whereClause = ""

        if (filter != "all" && filter != "recent") {
            if (filter.startsWith("level")) {
                val levelNum = filter.replace("level", "").toIntOrNull()
                if (levelNum != null) {
                    val categoryType = levelNum - 1
                    whereClause = "WHERE c.CategoryType = ?"
                    args.add(categoryType.toString())
                }
            }
        }
        
        if (whereClause.isNotEmpty()) {
            query += " $whereClause"
        }

        query += " ORDER BY a.Date DESC"

        val cursor = db.rawQuery(query, args.toTypedArray())
        
        cursor.use {
            while (it.moveToNext()) {
                val aid = it.getString(0)
                val dateFull = it.getString(1)
                val date = if (dateFull.contains(" ")) dateFull.split(" ")[0] else dateFull
                val score = it.getFloat(2).toInt()
                val categoryName = it.getString(3) ?: "Unknown"
                val sentence = it.getString(4) ?: "Unknown"
                val categoryType = it.getInt(5)
                
                val levelStr = "Level ${categoryType + 1}"
                val title = "$levelStr - $categoryName"
                
                items.add(HistoryUiItem(aid, date, title, sentence, score))
            }
        }
        
        return items
    }

    private fun cursorToPhoneHistoryItems(cursor: Cursor): List<PhoneHistoryItem> {
        val items = mutableListOf<PhoneHistoryItem>()
        cursor.use {
            while (it.moveToNext()) {
                val date = it.getString(it.getColumnIndexOrThrow("Date"))
                val aid = it.getString(it.getColumnIndexOrThrow("AID"))
                val wordIdx = it.getInt(it.getColumnIndexOrThrow("WordIdx"))
                val phoneIdx = it.getInt(it.getColumnIndexOrThrow("PhoneIdx"))
                val phone = it.getString(it.getColumnIndexOrThrow("Phone"))
                val category = it.getString(it.getColumnIndexOrThrow("Category"))
                val score = it.getFloat(it.getColumnIndexOrThrow("Score"))
                
                items.add(
                    PhoneHistoryItem(
                        date = date,
                        aid = aid,
                        wordIdx = wordIdx,
                        phoneIdx = phoneIdx,
                        phone = phone,
                        category = category,
                        score = score
                    )
                )
            }
        }
        
        return items
    }



    private fun cursorToWordHistoryItems(cursor: Cursor): List<WordHistoryItem> {
        val items = mutableListOf<WordHistoryItem>()
        cursor.use {
            while (it.moveToNext()) {
                val date = it.getString(it.getColumnIndexOrThrow("Date"))
                val aid = it.getString(it.getColumnIndexOrThrow("AID"))
                val wordIdx = it.getInt(it.getColumnIndexOrThrow("WordIdx"))
                val word = it.getString(it.getColumnIndexOrThrow("Word"))
                val numPhone = it.getInt(it.getColumnIndexOrThrow("NumPhone"))
                val color = it.getString(it.getColumnIndexOrThrow("Color"))
                val score = it.getFloat(it.getColumnIndexOrThrow("Score"))
                val comment = it.getString(it.getColumnIndexOrThrow("Comment"))
                
                items.add(
                    WordHistoryItem(
                        date = date,
                        aid = aid,
                        wordIdx = wordIdx,
                        word = word,
                        numPhone = numPhone,
                        color = color,
                        score = score,
                        comment = comment
                    )
                )
            }
        }
        
        return items
    }




    
    data class ContentItem(
        val uid: Int,
        val categoryType: Int,
        val contentsId: Int,
        val sentenceId: Int,
        val categoryName: String,
        val engLabel: String,
        val korLabel: String,
        val labelAudioSize: Int
    )

    data class AttemptItem(
        val date: String,
        val aid: String,
        val uid: Int,
        val numWord: Int,
        val transcript: String,
        val recordPath: String,
        val fluency: Float,
        val pronounce: Float
    )

    data class WordHistoryItem(
        val date: String,
        val aid: String,
        val wordIdx: Int,
        val word: String,
        val numPhone: Int,
        val color: String,
        val score: Float,
        val comment: String
    )

    data class PhoneHistoryItem(
        val date: String,
        val aid: String,
        val wordIdx: Int,
        val phoneIdx: Int,
        val phone: String,
        val category: String,
        val score: Float
    )

    data class HistoryUiItem(
        val aid: String,
        val date: String,
        val lessonTitle: String,
        val sentence: String,
        val score: Int
    )

    data class WeakPhoneItem(
        val phone: String,
        val category: String,
        val score: Int
    )
}
