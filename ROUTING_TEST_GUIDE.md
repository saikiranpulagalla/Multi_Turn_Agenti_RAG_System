# 🧪 Quick Test Guide - Routing & Memory

**How to verify all fixes are working**

---

## Test 1: Basic Web Routing ✅

**Do this**:
1. Make sure no PDFs are uploaded
2. Type: `"hello"` in chat
3. Click Send

**Expected to see**:
- ✅ Route badge shows: `🌐 Web`
- ✅ Response appears
- ✅ Metadata: `0 doc chunks • 0 web • Xms`

**What was wrong**: Previously showed `🔀 Hybrid` instead of `🌐 Web`

---

## Test 2: Document Upload & RAG Routing ✅

**Do this**:
1. Click "Upload a PDF document"
2. Choose any PDF file (even a small test PDF)
3. Wait for green checkmark ✅ "Document ingested"
4. Type: `"what is in this document?"` 
5. Click Send

**Expected to see**:
- ✅ Route badge shows: `🗂️ RAG`
- ✅ Response with document analysis
- ✅ Metadata shows: `X doc chunks • 0 web • Xms`

**What was wrong**: Previously showed route as `🔀 Hybrid` even with document question

---

## Test 3: Hybrid Routing with Document ✅

**Do this**:
1. Make sure document is still uploaded
2. Type: `"compare this document with current trends"`
3. Click Send

**Expected to see**:
- ✅ Route badge shows: `🔀 Hybrid`
- ✅ Response compares document + current info
- ✅ Metadata shows: `X doc chunks • 0 web • Xms`

**What was wrong**: Hybrid was being used inappropriately before

---

## Test 4: Document Memory Clearing ✅ (IMPORTANT)

**This is the key test**:

**Step 1**: Upload **PDF1** (e.g., "Notes.pdf")
```
1. Click "Upload a PDF document"
2. Upload: "Notes.pdf"
3. Wait for ✅ confirmation
4. Sidebar shows: ✓ Notes.pdf
```

**Step 2**: Ask about PDF1
```
1. Type: "Summarize this document"
2. Click Send
3. Look at response - it should mention specific content from PDF1
4. Note what it says
```

**Step 3**: Click "Clear Session" button
```
1. Click "🔄 Clear Session" in sidebar
2. Wait for: ✅ "Session cleared & documents removed"
3. Page should refresh
4. Sidebar should show: "No documents ingested yet"
5. Chat history should clear
```

**Step 4**: Upload **PDF2** (different file, e.g., "Report.pdf")
```
1. Click "Upload a PDF document"
2. Upload: "Report.pdf" (different from PDF1!)
3. Wait for ✅ confirmation
4. Sidebar shows: ✓ Report.pdf (NOT Notes.pdf)
```

**Step 5**: Ask about PDF2
```
1. Type: "Summarize this document"
2. Click Send
3. Look at response - it should be DIFFERENT from Step 2
4. It should reference PDF2 content only
```

**✅ If response in Step 5 is different from Step 2**: FIX WORKING!  
**❌ If response in Step 5 is SAME as Step 2**: Old memory persisting (bug)

---

## Test 5: Web Search Still Works ✅

**Do this**:
1. Click "Clear Session" (to have no documents)
2. Type: `"What was the latest technology news in 2024?"`
3. Click Send

**Expected to see**:
- ✅ Route badge shows: `🌐 Web`
- ✅ Response about 2024 tech news
- ✅ Metadata shows: `0 doc chunks • 0 web • Xms`

**Note**: Web results show 0 because SERP_API_KEY not set (that's OK)

---

## Quick Check List

- [ ] Test 1: Web routing works (`🌐 Web` badge)
- [ ] Test 2: RAG routing works (`🗂️ RAG` badge)
- [ ] Test 3: Hybrid routing works (`🔀 Hybrid` badge)
- [ ] Test 4a: Upload PDF1 and ask about it
- [ ] Test 4b: Clear session removes document
- [ ] Test 4c: Upload PDF2 shows different content
- [ ] Test 5: Web search still shows correct badge

**If all ✅**: All fixes working perfectly!

---

## Chat Messages You Should See

### After uploading document and asking about it:
```
👤 You
What's in this document?

🤖 Assistant 🗂️ RAG
Based on the document: [specific content from PDF]
Sources: 5 doc chunks • 0 web • 1,234ms
```

### After clearing and uploading new document:
```
👤 You
Summarize this document

🤖 Assistant 🗂️ RAG
Based on the document: [NEW document content, NOT old one]
Sources: 3 doc chunks • 0 web • 890ms
```

**The content should be COMPLETELY DIFFERENT!**

---

## Troubleshooting

**Q: Still seeing `🔀 Hybrid` for web queries?**  
A: Refresh browser (Ctrl+R) to clear cache

**Q: PDF not uploading?**  
A: Make sure file is actually a PDF, and size is reasonable

**Q: After clearing, still showing old document content?**  
A: That's the bug we fixed! Refresh browser and try again

**Q: Response says "No relevant information found"?**  
A: Normal for stub responses when SERP_API_KEY not set

---

## Expected Route Badges

| Query | Have Doc? | Badge | Route |
|-------|-----------|-------|-------|
| "hi" | No | 🌐 | Web |
| "what's this?" | Yes | 🗂️ | RAG |
| "compare" | Yes | 🔀 | Hybrid |
| "latest news" | Either | 🌐 | Web |
| "analyze" | Yes | 🗂️ | RAG |

---

*All tests should pass! System is now working correctly.* ✅
