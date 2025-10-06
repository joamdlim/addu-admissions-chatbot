#!/usr/bin/env node

/**
 * Simple test script to verify the guided conversation frontend works
 * Run this after starting both backend and frontend servers
 */

const https = require("http");

const BASE_URL = "http://127.0.0.1:8000/chatbot";

async function testEndpoint(url, method = "GET", data = null) {
  return new Promise((resolve, reject) => {
    const options = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
    };

    const req = https.request(url, options, (res) => {
      let body = "";
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => {
        try {
          const parsed = JSON.parse(body);
          resolve({ status: res.statusCode, data: parsed });
        } catch (e) {
          resolve({ status: res.statusCode, data: body });
        }
      });
    });

    req.on("error", reject);

    if (data) {
      req.write(JSON.stringify(data));
    }

    req.end();
  });
}

async function runTests() {
  console.log("🚀 Testing ADDU Guided Conversation Frontend");
  console.log("=".repeat(50));

  try {
    // Test 1: Get topics
    console.log("\n1️⃣ Testing GET /chatbot/topics/");
    const topicsResult = await testEndpoint(`${BASE_URL}/topics/`);

    if (topicsResult.status === 200) {
      console.log(
        `✅ Topics loaded: ${topicsResult.data.topics?.length || 0} topics`
      );
      topicsResult.data.topics?.slice(0, 3).forEach((topic) => {
        console.log(`   - ${topic.label} (${topic.id})`);
      });
    } else {
      console.log(`❌ Topics failed: ${topicsResult.status}`);
      return;
    }

    // Test 2: Initial guided chat request
    console.log("\n2️⃣ Testing initial guided chat request");
    const initialResult = await testEndpoint(
      `${BASE_URL}/chat/guided/`,
      "POST",
      {
        user_input: "",
        action_type: "message",
        session_id: "test_session_123",
      }
    );

    if (initialResult.status === 200) {
      console.log(`✅ Initial request successful`);
      console.log(`   State: ${initialResult.data.state}`);
      console.log(`   Input enabled: ${initialResult.data.input_enabled}`);
      console.log(`   Buttons: ${initialResult.data.buttons?.length || 0}`);
    } else {
      console.log(`❌ Initial request failed: ${initialResult.status}`);
    }

    // Test 3: Topic selection
    console.log("\n3️⃣ Testing topic selection");
    const topicResult = await testEndpoint(`${BASE_URL}/chat/guided/`, "POST", {
      user_input: "",
      action_type: "topic_selection",
      action_data: "admissions_enrollment",
      session_id: "test_session_123",
    });

    if (topicResult.status === 200) {
      console.log(`✅ Topic selection successful`);
      console.log(`   State: ${topicResult.data.state}`);
      console.log(`   Current topic: ${topicResult.data.current_topic}`);
      console.log(`   Input enabled: ${topicResult.data.input_enabled}`);
      console.log(
        `   Response: ${topicResult.data.response?.substring(0, 100)}...`
      );
    } else {
      console.log(`❌ Topic selection failed: ${topicResult.status}`);
    }

    // Test 4: Question within topic
    console.log("\n4️⃣ Testing question within topic");
    const questionResult = await testEndpoint(
      `${BASE_URL}/chat/guided/`,
      "POST",
      {
        user_input: "What are the admission requirements?",
        action_type: "message",
        session_id: "test_session_123",
      }
    );

    if (questionResult.status === 200) {
      console.log(`✅ Question processing successful`);
      console.log(`   State: ${questionResult.data.state}`);
      console.log(`   Sources: ${questionResult.data.sources?.length || 0}`);
      console.log(
        `   Response: ${questionResult.data.response?.substring(0, 150)}...`
      );
    } else {
      console.log(`❌ Question processing failed: ${questionResult.status}`);
    }

    console.log("\n✅ All tests completed!");
    console.log("\n📋 Frontend Checklist:");
    console.log(
      "   1. Start backend: cd backend && python manage.py runserver"
    );
    console.log("   2. Start frontend: cd frontend && npm run dev");
    console.log("   3. Open browser: http://localhost:5173");
    console.log("   4. Test guided conversation flow");
    console.log("   5. Toggle between Guided and Free Chat modes");
  } catch (error) {
    console.error("❌ Test failed:", error.message);
    console.log("\n🔧 Troubleshooting:");
    console.log("   - Make sure backend is running on port 8000");
    console.log("   - Check if Django server is accessible");
    console.log("   - Verify CORS settings allow frontend requests");
  }
}

if (require.main === module) {
  runTests();
}

module.exports = { testEndpoint, runTests };
