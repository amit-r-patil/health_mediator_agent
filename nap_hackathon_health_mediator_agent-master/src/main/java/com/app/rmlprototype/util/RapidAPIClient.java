package com.app.rmlprototype.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.DefaultHttpClient;
import org.json.JSONObject;

public class RapidAPIClient {

    private static String RAPID_SMS_URL = "https://rapidapi.rmlconnect.net:9443/bulksms/bulksms?username=rapid-KTXa3304110000&password=617bf204245383001100f813&type=0&dlr=0&source=RMLPRD";

    private static String RAPID_EMAIL_URL = "https://rapidemail.rmlconnect.net/v1.0/messages/sendMail";

    private static String RAPID_WHATS_APP_URL = "https://rapidapi.rmlconnect.net/wbm/v1/message";

    private static final String USER_AGENT = "Mozilla/5.0";

    public static String sendSMS(String toNumber, String message) {

        String returnString = "Sorry.. Failed to Send SMS Notification";
        try {
            String requestUrl = RAPID_SMS_URL + "&destination=" + toNumber + "&message=" + URLEncoder.encode(message, "utf-8");
            System.out.println(requestUrl);
            DefaultHttpClient httpClient = new DefaultHttpClient();
            HttpGet getRequest = new HttpGet(requestUrl);

            HttpResponse response = httpClient.execute(getRequest);


            if (response.getStatusLine().getStatusCode() == 200) {
                BufferedReader br = new BufferedReader(
                        new InputStreamReader((response.getEntity().getContent())));

                String output;
                StringBuffer responseLine = new StringBuffer();
                System.out.println("Output from Server .... \n");
                while ((output = br.readLine()) != null) {
                    System.out.println(output);
                    responseLine.append(output);
                }

                if (responseLine.toString().startsWith("1701")) {
                    returnString = "SMS Sent Sucessfully on " + toNumber;
                }
            }


            httpClient.getConnectionManager().shutdown();


        } catch (IOException e) {
            e.printStackTrace();
        }
        return returnString;
    }


    public static String sendEmail(String toEmail, String toName, String subject, String message, Map<String, String> attachments) {
        String resultResponse = "Sorry.. Failed to send email notification.";

        try {
            DefaultHttpClient httpClient = new DefaultHttpClient();
            HttpPost postRequest = new HttpPost(RAPID_EMAIL_URL);

            Map<String, Object> inputMap = new HashMap<String, Object>();
            inputMap.put("owner_id", "40307286");
            inputMap.put("token", "rrjZmtORxLf4HzWWfcmmlQGX");
            inputMap.put("smtp_user_name", "smtp76217062");

            Map<String, Object> messageMap = new HashMap<String, Object>();
            messageMap.put("text", message);
            messageMap.put("subject", subject);
            messageMap.put("from_email", "noreply@rapidemail.rmlconnect.net");
            messageMap.put("from_name", "RAPID HMA");

            Map<String, String> toMap = new HashMap<String, String>();
            toMap.put("email", toEmail);
            toMap.put("name", toName);
            toMap.put("type", "to");
            messageMap.put("to", toMap);

            Map<String, String> headersMap = new HashMap<String, String>();
            headersMap.put("Reply-To", "message.reply@example.com");
            headersMap.put("X-Unique-Id", "register_"+toEmail);
            messageMap.put("headers", headersMap);

            List<Map<String, String>> attachmentsList = new ArrayList<Map<String,String>>();
            if(attachments!= null && attachments.size()> 0)
            {
                for (String attachment : attachments.keySet()) {
                    Map<String, String> attachmentMap = new HashMap<String, String>();

                    System.out.println(attachment);
                    String[] fileSplits = attachment.toLowerCase().split("\\.");
                    String extension = fileSplits[fileSplits.length-1];

                    if (extension.equals("pdf"))
                    {
                        attachmentMap.put("type", "text/pdf");
                        byte[] fileContent = FileUtils.readFileToByteArray(new File(attachments.get(attachment)));
                        String encodedString = Base64.getEncoder().encodeToString(fileContent);
                        attachmentMap.put("content", encodedString);
                    }
                    else if (extension.equals(".png"))
                    {
                        attachmentMap.put("type", "text/png");
                        byte[] fileContent = FileUtils.readFileToByteArray(new File(attachments.get(attachment)));
                        String encodedString = Base64.getEncoder().encodeToString(fileContent);
                        attachmentMap.put("content", encodedString);
                    }
                    else if(extension.equals(".jpeg"))
                    {
                        attachmentMap.put("type", "text/jpeg");
                        byte[] fileContent = FileUtils.readFileToByteArray(new File(attachments.get(attachment)));
                        String encodedString = Base64.getEncoder().encodeToString(fileContent);
                        attachmentMap.put("content", encodedString);
                    }
                    else
                    {
                        attachmentMap.put("type", "text/jpg");
                        byte[] fileContent = FileUtils.readFileToByteArray(new File(attachments.get(attachment)));
                        String encodedString = Base64.getEncoder().encodeToString(fileContent);
                        attachmentMap.put("content", encodedString);
                    }


                    attachmentMap.put("name", attachment);

                    attachmentsList.add(attachmentMap);
                }
            }
            messageMap.put("attachments", attachmentsList);

            inputMap.put("message", messageMap);



            JSONObject json = new JSONObject(inputMap);
            StringEntity input = new StringEntity(json.toString());
            //StringEntity input = new StringEntity(inputMap.toString());
            input.setContentType("application/json");
            postRequest.setEntity(input);

            HttpResponse response = httpClient.execute(postRequest);

            if (response.getStatusLine().getStatusCode() == 200) {
                resultResponse = "Email sent seuccessfully to " + toEmail;
            }

            httpClient.getConnectionManager().shutdown();


        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(resultResponse);
        return resultResponse;
    }

    public static String sendWhatsAppDocument(String toNumber, String fileUrl, String caption, String filename)
    {
        String resultResponse = "Sorry.. Failed to send Whats App Document.";

        try {
            String fileType = "document";
            if(filename.endsWith(".jpg") || filename.endsWith(".jepg") || filename.endsWith(".png")) {
                fileType = "image";
            }

            DefaultHttpClient httpClient = new DefaultHttpClient();
            HttpPost postRequest = new HttpPost(RAPID_WHATS_APP_URL);

            Map<String, Object> inputMap = new HashMap<String, Object>();
            inputMap.put("phone", toNumber);
            Map<String, String> mediaMap = new HashMap<String, String>();
            mediaMap.put("type", fileType);
            mediaMap.put("url", fileUrl);
            mediaMap.put("caption", caption);
            mediaMap.put("file", filename);

            inputMap.put("media", mediaMap);


            JSONObject json = new JSONObject(inputMap);
            StringEntity input = new StringEntity(json.toString());
            //StringEntity input = new StringEntity(inputMap.toString());
            input.setContentType("application/json");
            postRequest.setEntity(input);

            postRequest.setHeader("Authorization", "617bf204245383001100f813");
            HttpResponse response = httpClient.execute(postRequest);


            if (response.getStatusLine().getStatusCode() == 202) {
                resultResponse = "Whats App Document sent successfully to " + toNumber;
            }

            httpClient.getConnectionManager().shutdown();


        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(resultResponse);
        return resultResponse;
    }

    public static void main(String args[]) {
        //System.out.println(RapidAPIClient.sendSMS("+919226161589", "Hello Again Test message...1"));
        //Map<String, String> attachments = new HashMap<String, String>();
        //attachments.put("apc.jpg", "/Users/amipatil/Downloads/3.jpg");
        //attachments.put("xyz.pdf", "/Users/amipatil/Downloads/myfile.pdf");
        //sendEmail("patil1995amit@gmail.com", "Amit P.1", "Hello Subject : Testing from Java 1",
        //"Hello Body : Testing from Java 1", null);
        RapidAPIClient.sendWhatsAppDocument("+919226161589", "https://i.postimg.cc/C19KsW-gd/fracture1.jpg", "Fractured bone detected", "fractured_bone.jpg");

    }
}
